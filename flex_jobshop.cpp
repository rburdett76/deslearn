#include "flex_jobshop.h"
//#include "simulator.cpp" // May be needed here 

#include <cmath>


const int MEAN = 0;
const int VAR = 1;

extern int NTHRDS;

namespace FlexJobShop {

    std::uniform_int_distribution<> pt_short(1, 30); // minutes
    std::uniform_int_distribution<> pt_med(30, 60); // minutes
    std::uniform_int_distribution<> pt_long(60, 120); // minutes
    std::uniform_int_distribution<> type(1, 3); // long, med, short
    std::uniform_int_distribution<> flex(1, 3); // 1,2,or 3 candidate machines
    std::uniform_real_distribution<> unif_0_1(0, 1);

    /* Remove 1 character from position 0 */
    string discard_first(string str) { return str.erase(0, 1); }

    /* Data */
    SimTime choose_pt(int type, std::mt19937& gen) {
        switch (type) {
        case 1: return (SimTime)pt_short(gen);
        case 2: return (SimTime)pt_med(gen);
        case 3: return (SimTime)pt_long(gen);
        default: throw std::invalid_argument("choose_pt: invalid input");
        }
    }

    /* Queue handling - q is a copy */
    vector<Activity*> extract_queue(queue<Activity*> q, std::mt19937* gen) {
        vector<Activity*> lst;
        while (!q.empty()) { lst.push_back(q.front()); q.pop(); }
        if (gen != nullptr) std::shuffle(lst.begin(), lst.end(), *gen);
        return lst;
    }

    Machine* as_machine(SimObject* obj) { return dynamic_cast<Machine*>(obj); }
    Activity* as_activity(SimObject* obj) { return dynamic_cast<Activity*>(obj); }
    Job* as_job(SimObject* obj) { return dynamic_cast<Job*>(obj); }

    /* Casting base objects */
    bool is_machine(SimObject* obj, Machine* &machine) {
        machine = dynamic_cast<Machine*>(obj);
        return (machine!= nullptr);
    }

    /* Casting base objects */
    bool is_job(SimObject* obj, Job* &job) {
        job = dynamic_cast<Job*>(obj);
        return (job != nullptr);
    }
    
    /* Casting base objects */
    bool is_activity(SimObject* obj, Activity* &activity) { 
        activity = dynamic_cast<Activity*>(obj);
        return (activity != nullptr);
    }
    
    bool is_complete(ObjState _state) { return (_state == "COMPLETE"); }
    bool not_complete(ObjState _state) { return !is_complete(_state); }
    bool is_pending(ObjState _state) { return (_state == "PENDING"); }
    bool in_progress(ObjState _state) { return (_state == "IN_PROGRESS"); }
    bool is_idle(ObjState _state) { return (_state == "IDLE"); }
    bool is_busy(ObjState _state) { return (_state == "BUSY"); }

    /*********************************************************/
    /* ----------------- RESULTS CLASS --------------------- */
    /*********************************************************/

    Metrics::Metrics() { 
        score = &Metrics::get_cmax; // Set makespan as the main KPI
        makespan = (SimTime)0;
        flowtime = (SimTime)0;
        stdev_makespan = (SimTime)0;
        stdev_flowtime = (SimTime)0;
    }

    /*********************************************************/
    /* ----------------- ACTIVITY CLASS -------------------- */
    /*********************************************************/

    /* Constructor */
    Activity::Activity(string _descr, Sim& _sim, Job& _job, int _num)
        :SimObject(_descr, _sim), job(_job), num(_num) {
        start = end = -1;
        selected_pt = nullptr;
        set_state("PENDING");        
    }

    Activity::~Activity() {
        for (auto &pt : pts) delete pt.second;
        pts.clear();
        options.clear();
        selected_pt = nullptr;
    }

    /* Copying */
    void Activity::copy(Activity& source) {     
        for (auto& _pt : source.pts) {
            Parameter *par = new Parameter(job.shop.gen);           
            par->copy(*_pt.second);
            pts.insert({ _pt.first, par });
        }            
         
        for (auto& option : source.options) {           
            Machine* mach = job.shop.machines[option->num];
            add_option(mach, pts[mach->descr]);
        }                
    }

    /* Initialisation */
    void Activity::add_option(Machine* _mach, Parameter *pt) {
        options.push_back(_mach);
        pts[_mach->descr] = pt;
    }

    /* Logging */
    void Activity::log_details(ostream& os) {
        os << "\n[" << descr << "]:";
        os << " cand=[";
        bool first = true;
        for (auto option : options) {
            if (!first) os << ","; else first = false;
            os << option->descr << "(" << pts[option->descr] << ")";            
        }
        os << "]";
    }

    /* State */
    SimTime Activity::get_remain(SimTime t) {    
        if (is_complete(sim_state)) return 0;
        else if(is_pending(sim_state)) return selected_pt->sample<SimTime>();
        else if(in_progress(sim_state))  return (selected_pt->sample<SimTime>() - (t - start));
        return -1;
    }

    /* */
    void Activity::assign_machine(Machine* _machine) {
        assert(_machine != nullptr);
        machine = _machine; // Record machine assigned
        selected_pt = pts[_machine->descr];     
    }

    /* */
    SimTime Activity::get_pt(Machine* _machine, bool sample) { 
        Parameter* par = pts[_machine->descr];
        if (sample)
            return par->sample<SimTime>();
        else             
            return par->mean<SimTime>();        
    }

    /* Access Functions */
    Activity* Activity::succ() { return job.succ(this); }
    Activity* Activity::pred() { return job.pred(this); }

    /* Process */
    SimCoroutine Activity::run() { co_return; } // No process is modelled

    /*********************************************************/
    /* --------------------- JOB CLASS --------------------- */
    /*********************************************************/

    /* Constructor */
    Job::Job(string _descr, int _num, JobShop& _shop, Sim& _sim) 
        :descr(_descr), num(_num), shop(_shop), SimObject(_descr, _sim) {
        current = nullptr;
        rlt = 0;
        set_state("PENDING");
    }

    /* Destructor */
    Job::~Job() {
        for (auto activity : activities) delete activity;
    }

    /* Copying */
    void Job::copy(Job& source) {
        rlt = source.rlt;
        for (auto& activity : source.activities) {
            Activity* new_activity = new Activity(activity->descr, sim, *this, activity->num); 
            new_activity->copy(*activity);
            add_activity(*new_activity);                        
        }
    }
    
    /* Initialisation */
    void Job::add_activity(Activity& _activity) {
        activities.push_back(&_activity);
        shop.activities.push_back(&_activity); // Update central listing
        _activity.id = (int) (shop.activities.size()) - 1;
    }
    
    /* */
    void Job::set_current(Activity* activity) { current = activity;  }

    /* Access Functions */
    Activity* Job::first() { return activities.front(); }
    Activity* Job::last() { return activities.back(); }

    /* Access Function */
    Activity* Job::succ(Activity* activity) {
        int index = activity->num + 1;
        Activity* act = (index <= activities.size() - 1) ? activities[index] : nullptr;
        return act;
    }

    /* Access Function */
    Activity* Job::pred(Activity* activity) {
        int index = activity->num - 1;
        Activity* act = (index >= 0) ? activities[index] : nullptr;
        return act;
    }
 
    /* Logging */
    void Job::log_details(ostream& os) {
        os << "\n[" << descr << "]: rlt=" << rlt << ";";
        for (Activity* activity : activities) activity->log_details(os);
        os << "\n";
    }
    
    /* Process */
    SimCoroutine Job::run() { co_return; } // No process is modelled

    /*********************************************************/
    /* ----------------- MACHINE CLASS --------------------- */
    /*********************************************************/

    /* Constructor */
    Machine::Machine(string _descr, JobShop& _shop, Sim &_sim)
        :descr(_descr), shop(_shop), SimObject(_descr, _sim) {
        set_state("IDLE");
    }

    SimCoroutine Machine::run() { co_return; }  // No process is modelled

    /*********************************************************/
    /* ---------------- DISPATCHER CLASS ------------------- */
    /*********************************************************/
    
    /* Constructor */
    Dispatcher::Dispatcher(std::string _descr, Sim& _sim, JobShop &_shop):SimObject(_descr, _sim), shop(_shop) {
        random_choice = true; 
        set_state("OPERATING");     
        condition = new SimCondWaiting(sim, [this]() { return is_ready(); });
        condition->is_dp = true; // When the conditon is triggered a decision must be made
        activate(); // Activate a process        
    }   

    /* Check which candidate machine is free at this moment in time */
    Machine* Dispatcher::choose_machine(Activity* activity) {
        free.clear();
        for (Machine* machine : activity->options) {
            if(is_idle(machine->sim_state)) free.push_back(machine);
        }
        if (free.empty()) return nullptr; // No free machine
        /*_________________________________________________*/
        if (random_choice) {
            std::uniform_int_distribution<> choice(0, static_cast<int>(free.size()) - 1);
            return free[choice(shop.gen)];
        }
        else return free[0];// Choose first free machine
    }

    bool Dispatcher::is_ready() { 
        return (!shop.ready_to_process.empty()); 
    }

    /* Dispatching process */
    bool Dispatcher::process(Activity& activity) {
        Machine* mach = choose_machine(&activity);
        if (mach != nullptr) {
            activity.assign_machine(mach);
            mach->set_state("BUSY"); // This ensures the machine is not assigned again before the scheduled commencment event is handled
            sim.schedule_event(sim.now(), "COMMENCE", &activity, shop.manager);
            shop.dequeue_activity(&activity); // Remove activity from ready_to_process queue
            shop.record_assignment(sim.now(), &activity, activity.machine); // Revise system history 
            return true;
        }
        return false;
    }

    /* Assumption: If an activity is ready, assign it to a ready machine [non-delay policy] */
    void Dispatcher::process_ready_activity() {        
        vector<Activity*> ready = extract_queue(shop.ready_to_process, &shop.gen);
        for (auto it = ready.begin(); it != ready.end(); ++it) process(**it);        
    }

    /* Assumption: Only called if ready.size() > 0 */
    void Dispatcher::process_next_ready_activity() {
        vector<Activity*> ready = extract_queue(shop.ready_to_process, &shop.gen);
        for (Activity* activity : ready) {
            if (process(*activity)) return;            
        }
    }

    /* */
    void Dispatcher::get_possible_actions(vector<Action>& actions) {
        vector<Activity*> ready = extract_queue(shop.ready_to_process, &shop.gen);
        for (auto activity: ready) {
            for (Machine* machine : activity->options) {
                if (is_idle(machine->sim_state)) {
                    string code = activity->descr + "_"  + machine->descr;
                    int a = shop.find_action(code);
                    Action action; 
                    action.define(a);
                    actions.push_back(action);
                }
            }
        }
    }


    /* Dispatching process */
    SimCoroutine Dispatcher::run() {
        while (true) {
           co_await await_condition(*condition); // Wait until this condition is met                         
           
           /// SYSTEM LOGIC
           process_next_ready_activity(); 
           
           /// EXTERNAL DECISION MAKER PORTAL
           /*SimTime t = sim.now();
           SystemState state = shop.extract_state(t);
           int a = sim.decision_interface->request_action(state, t);
           Action action = shop.copy_action(a);
           shop.apply_action(state, action);    */     
           
           co_await suspend_execution(); // Yield control back to the caller
        }        
        co_return; // Terminate the coroutine
    }    


    /*********************************************************/
    /* ----------------- MANAGER CLASS --------------------- */
    /*********************************************************/

    /* */
    OperationsManager::OperationsManager(std::string _descr, Sim& _sim, JobShop& _shop) :SimObject(_descr, _sim), shop(_shop) {
        set_state("OPERATING");
        activate(); // Activate a process        
    }

    /* */
    SimCoroutine OperationsManager::run() {
        co_await suspend_execution(); // Suspend execution and yield control back to the caller
        while (true) {
            SimQueuedEvent* qe = sim.next_event;
            Activity* activity = dynamic_cast<Activity*>(sim.next_event->obj);
            
            if (sim.next_event->type == "READY") {
                //cout << "\nOPERATIONS MANAGER::ENQUEUE " << activity->descr << " @t= " << sim.now();
                shop.enqueue_activity(activity);
            }

            if (sim.next_event->type == "COMMENCE") {
                //cout << endl << "OPERATIONS MANAGER::COMMENCE " << activity->descr << " @t= " << sim.now();
                activity->set_state("IN_PROGRESS");
                activity->start = sim.now();                
                if (activity->num == 0) activity->job.set_state("IN_PROGRESS"); // Revise job state
                SimTime pt = activity->selected_pt->sample<SimTime>(); // OR USE: activity->get_pt(activity->machine, sample) (?)        
                sim.schedule_event(sim.now() + pt, "COMPLETION", activity, shop.manager);
            }

            if (sim.next_event->type == "COMPLETION") {
                //cout << endl << "OPERATIONS MANAGER:: COMPLETE " << activity->descr << " @t= " << sim.now();
                activity->set_state("COMPLETE");
                activity->end = sim.now(); // Record completion time                  
                activity->machine->set_state("IDLE"); // Revise state
                activity->job.set_current(activity->succ());
                if (!activity->job.current) 
                    activity->job.set_state("COMPLETE");
                else 
                    shop.enqueue_activity(activity->job.current);
            }   

            co_await std::suspend_always();
        }
        co_return;
    }

    /*********************************************************/
    /* ---------------- JOBSHOP CLASS ---------------------- */
    /*********************************************************/

    /* Constructor */
    JobShop::JobShop(Sim &_sim):System(_sim) {
        dispatcher = new Dispatcher("dispatcher", _sim, *this);
        manager = new OperationsManager("manager", _sim, *this);
        cmax = -1;
        _sim.system = this;
        reward = RewardType::incr; // default or RewardType::delay;
    }

    /* Destructor */
    JobShop::~JobShop() {
        for (auto machine : machines) delete machine;
        for (auto job : jobs) delete job;
        delete dispatcher; 
        delete manager;
        if (state_zero != nullptr) delete state_zero;        
    }

    /* */
    void JobShop::clear_queue() {
        while (!ready_to_process.empty()) ready_to_process.pop();
    }

    /* */
    void JobShop::enqueue_activity(Activity* act) { ready_to_process.push(act); }

    /* */
    void JobShop::dequeue_activity(Activity* activity) {
        std::queue<Activity*> temp;
        while (!ready_to_process.empty()) {
            if (ready_to_process.front() != activity) temp.push(ready_to_process.front());
            ready_to_process.pop();
        }
        while (!temp.empty()) { ready_to_process.push(temp.front()); temp.pop(); } // Rebuild the queue
    }

    /* Perform deep copy of machine, job and activity */
    void JobShop::copy(const JobShop& shop) { 

        for (auto& machine : shop.machines) {
            Machine* new_machine = new Machine(machine->descr, *this, sim);
            add_machine(new_machine);
        }

        for (auto& job : shop.jobs) {
            Job* new_job = new Job(job->descr, job->num, *this, sim);   
            new_job->copy(*job);
            add_job(new_job);
        }         
    }

    /* */
    void JobShop::copy(const System& system) {
        const JobShop* js = dynamic_cast<const JobShop*>(&system);
        if (!js) {
            throw std::runtime_error("System::copy called with non-JobShop source");
        }
        copy(*js);   // Call the real implementation
    }
    
    /* Clone the systems machines, jobs, activities and perform other intialisations */
    std::unique_ptr<System> JobShop::clone(Sim &_sim) const {
        // Note: auto == std::unique_ptr<JobShop>
        auto cloned = std::make_unique<JobShop>(_sim);        
        cloned->copy(*this);
        cloned->define_action_space();
        cloned->state_zero = cloned->create_initial_state();
        return cloned;
    }

    /* Set every machine as idle, and every job and job activity as pending */
    SystemState* JobShop::create_initial_state() {
        SystemState* state_zero = new SystemState();
   
        for (auto machine : machines) {
            StateRecord record(machine, "IDLE");
            state_zero->add(record);
        }

        for (auto job : jobs) {
            StateRecord record(job, "PENDING");
            state_zero->add(record);
            string str = job->descr + "|1";
            state_zero->add_more_info(job->descr, str);
            for (auto activity : job->activities) {
                StateRecord record(activity, "PENDING");
                state_zero->add(record);
            }
        }
        return state_zero;
    }

    /* Choose job priorities. Schedule first activity of each job */
    void JobShop::schedule_first_activity(Job *job) {
        job->set_current(job->first());
        float priority = (float) (1000*unif_0_1(gen));
        sim.schedule_event(job->rlt, "READY", job->current, manager, static_cast<int>(priority));
    }

    /* Reset the state of every SimObject in the JobShop */
    /* machines => ObjState::IDLE, jobs/activity => ObjState::PENDING */
    /* schedule first activity of each job */
    void JobShop::reset() { restore_state(*state_zero); }

    /* System Simulation */
    /* Assumption: reset has been called or a recorded historical state has been restored */
    void JobShop::simulate(SimResult& result) { System::simulate(result); }

    /* Measuring Performance */
    void JobShop::evaluate_schedule(Schedule &schedule, Metrics &metrics) {
        metrics.flowtime = 0;
        metrics.makespan = 0;
        for (Job* job : jobs) {
            tuple<SimTime, SimTime, int> ref_tuple = schedule[job->last()->descr];
            metrics.flowtime += get<1>(ref_tuple); // Add the completion time
            metrics.makespan = fmax(get<1>(ref_tuple), metrics.makespan);
        }
    }

    /* */
    void JobShop::evaluate_deter(PriorityOrder& order, Metrics& metrics) {
        Schedule *_schedule = build_schedule(order, false); // sample = false
        evaluate_schedule(*_schedule, metrics);
        delete _schedule;
    }

    /* */
    void JobShop::evaluate_stoch(PriorityOrder& order, Metrics& metrics, bool par_eval) {
        if (par_eval)
            evaluate_expected_performance_parallel(order, metrics, 1000);  
        else
            evaluate_expected_performance_stable(order, metrics, 0.01f);  
        
        // evaluate_expected_performance(order, metrics, 1000);
        // evaluate_expected_performance_stable(order, metrics, 0.001);
    }

    /* A schedule is built */
    void JobShop::evaluate(PriorityOrder& order, Metrics& metrics, bool stoch) {
        if (stoch) evaluate_stoch(order, metrics, true);         
        else evaluate_deter(order, metrics);        
    }


    /* Measure expected makespan and flowtime - faster parallel version */
   void JobShop::evaluate_expected_performance_parallel(PriorityOrder& order, Metrics& metrics, int n) {
      
       metrics.flowtime = 0.0;
       metrics.makespan = 0.0;

       float cmax = 0;

       std::vector<Metrics> thread_sums(NTHRDS, Metrics{});
       std::vector<int> thread_counts(NTHRDS, 0);

       #pragma omp parallel for num_threads(NTHRDS)
       for (int i = 0; i < n; ++i) {
           Schedule* schedule = build_schedule(order, true);
           Metrics local_metrics;
           evaluate_schedule(*schedule, local_metrics);
           int tid = omp_get_thread_num();
           thread_counts[tid]++;
           thread_sums[tid].makespan += local_metrics.makespan;
           thread_sums[tid].flowtime += local_metrics.flowtime;
           delete schedule;
       }

       // Aggregate sums and compute averages
       for (int thrd = 0; thrd < NTHRDS; thrd++) {
           metrics.flowtime += thread_sums[thrd].flowtime;
           metrics.makespan += thread_sums[thrd].makespan;
       }
       metrics.flowtime /= n;
       metrics.makespan /= n;
    }

    /* Measure expected makespan and flowtime - basic version */
    void JobShop::evaluate_expected_performance(PriorityOrder& order, Metrics& mean, int n) {
        mean.makespan = mean.flowtime = 0;
        for (int i = 0; i < n; i++) {
            Schedule* schedule = build_schedule(order, true);
            Metrics result;
            evaluate_schedule(*schedule, result);
            mean.makespan = mean.makespan + (result.makespan - mean.makespan) / (SimTime)(i + 1.0);
            mean.flowtime = mean.flowtime + (result.flowtime - mean.flowtime) / (SimTime)(i + 1.0);
            delete schedule;
        }
    }

    /* Measure expected makespan and flowtime - long version  */
    /* Terminate when stability reached - i.e., moments become invariant 
        Skewness: '=0': symmetric, '>0': right tail, '<0': left tail
        Kurtosis: '>3': more outliers, '=3': normal distr , '<3': fewer outliers
    */
    void JobShop::evaluate_expected_performance_stable(PriorityOrder &order, Metrics &result, float tol) {                
        Schedule* schedule;
        Metrics metrics;
        vector<SimTime> moments_makespan(2, 0); // (mean=0, var=0)
        vector<SimTime> moments_flowtime(2, 0);
        vector<SimTime> previous_moments_makespan(2, 0);
        SimTime diff, alpha, beta;
        int scenario_count = 0;
        bool continue_simulation;
        do {
            scenario_count++;
            schedule = build_schedule(order, true);
            evaluate_schedule(*schedule, metrics);
           
            if (scenario_count == 1) {
                moments_makespan[MEAN] = metrics.makespan;
                moments_flowtime[MEAN] = metrics.flowtime;
            }
            else {
                beta = 1.0f / scenario_count;
                alpha = 1.0f - beta;
                for (int i = 0; i < 2; i++) previous_moments_makespan[i] = moments_makespan[i]; // Record previous state

                diff = metrics.makespan - moments_makespan[MEAN];
                moments_makespan[MEAN] = moments_makespan[MEAN] + beta * diff; // Mean
                moments_makespan[VAR] = alpha * moments_makespan[VAR] + alpha * beta * (float) pow(diff, 2); // Var

                diff = metrics.flowtime - moments_flowtime[MEAN];
                moments_flowtime[MEAN] = moments_flowtime[MEAN] + beta * diff; // Mean
                moments_flowtime[VAR] = alpha * moments_flowtime[VAR] + alpha * beta * (float) pow(diff, 2); // Var
            }

            delete schedule;

            continue_simulation = false;
            for (int i = 0; i < 2; i++) {
                /// Measure change in metric 
                SimTime delta = abs(moments_makespan[i] - previous_moments_makespan[i]); 

                /// Change is too large => evaluate more scenarios
                if (delta > tol) { continue_simulation = true;  break; } 
            }

        } while (continue_simulation);

        // Package the results:
        result.makespan = moments_makespan[MEAN];
        result.stdev_makespan = moments_makespan[VAR];
        result.flowtime = moments_flowtime[MEAN];
        result.stdev_flowtime = moments_flowtime[VAR];
        result.nb_scen = scenario_count;       
    }

    /* Solution - Load an existing priority ordering from a file */
    void JobShop::load_order(string filename, PriorityOrder &order) {
        ifstream in(filename);
        if (in.is_open()) {
            string txt;
            vector<string> sub;
            while (getline(in, txt)) {
                sub = split_string(txt, '/');
                int j = atoi(discard_first(sub[0]).c_str()) - 1;
                int a = atoi(discard_first(sub[1]).c_str()) - 1;
                int m = atoi(discard_first(sub[2]).c_str()) - 1;
                pair<Activity*, Machine*>* pr = new pair<Activity*, Machine*>(jobs[j]->activities[a], machines[m]);
                order.push_back(*pr);
            }
        }
    }

    /* FJSP FORMAT. First line: <number of jobs> <number of machines>
    Then one line per job: <number of operations> and then, for each operation, 
    <number of machines for this operation> and for each machine, a pair <machine> <processing time>.
    Machine index starts at 0.*/
    void JobShop::load_instance(const string& filename) {
        ifstream in(filename);
        if (in.is_open())
        {
            int nJ, nM, nO, nMo, m;
            float pt;
            in >> nJ >> nM;
            create_machines(nM);
            create_jobs(nJ);
            for (int i = 0; i < nJ; i++) {
                in >> nO; // Number of job operations                
                for (int o = 0; o < nO; o++) {
                    vector<pair<int, Parameter*>> cand;
                    in >> nMo; // Number of candidate machines for operation o
                    for (int j = 0; j < nMo; j++) {
                        in >> m >> pt;
                        Parameter* par = new Parameter(gen, Distribution::fixed, vector<float> {pt});
                        assert(par != nullptr);
                        cand.push_back(pair<int, Parameter*>(m, par));
                    }
                    define_activity(i, o, cand);
                }
            }
            in.close();
        }
        else {
            throw std::runtime_error("File could not be found!");
            exit(1);
        }
        define_action_space();
    }

    /* */
    Distribution get_distr(string str) {
        if (str == "Unif") return Distribution::uniform;
        if (str == "Fixed") return Distribution::fixed;
        if (str == "Tri") return Distribution::triangular;
        if (str == "Norm") return Distribution::normal;
        if (str == "LogNorm") return Distribution::log_normal;

        // Handle unknown input if necessary
        throw std::invalid_argument("Unknown distribution type");
    }

    /* Format: distr(par1,par2...) */
    Parameter* load_random_value(string str, std::mt19937& gen) {
        vector<string> sub = split_string(str, '(');   
        Parameter* rv = new Parameter(gen);
        rv->distr = get_distr(sub[0]);
        switch (rv->distr) {
            case Distribution::fixed: break;
            case Distribution::uniform: break;
            case Distribution::normal: break;
            case Distribution::triangular: break;
        }
        rv->control.clear();
        sub[1].pop_back(); // Remove ")"
        vector<string> param = split_string(sub[1], ',');
        for (int i = 0; i < (int) param.size(); i++) {
            float value = (float)atof(param[i].c_str());
            rv->control.push_back(value);
        }

        return rv;
    }

    /* */
    void JobShop::load_stoch_instance(const string& filename) {
        ifstream in(filename);
        if (in.is_open())
        {
            string txt;
            int nJ, nM, job_id, act_id, mach_id;
            vector<string> sub, rv;

            getline(in, txt);
            sub = split_string(txt, ',');
            nJ = atoi(sub[0].c_str());
            nM = atoi(sub[1].c_str());
            create_machines(nM);
            create_jobs(nJ);

            while (getline(in, txt)) {
                sub = split_string(txt, '/');
                job_id = atoi(discard_first(sub[0]).c_str()) - 1;
                act_id = atoi(discard_first(sub[1]).c_str()) - 1;
                vector<pair<int, Parameter*> > cand;
                for (int i = 2; i < (int) sub.size(); i += 2) {
                    mach_id = atoi(discard_first(sub[i]).c_str()) - 1;
                    Parameter* rv = load_random_value(sub[i + 1], gen); // Load random value
                    cand.push_back(pair<int, Parameter*>(mach_id, rv));
                }
                define_activity(job_id, act_id, cand);
            }
            define_action_space();
        }
        else {
            throw std::runtime_error("File could not be found!");
            exit(1);
        }
    }

    // Customise distribution for each processing time. 
    // Assumption: Starting distribution is "::Fixed"
    void JobShop::customize_instance(const string& filename) {
        for (auto activity : activities) {
            for(auto option: activity->options) // These are machines!
            {    
                cout << "\nRevising "<<activity->descr<<"/"<<option->descr<<"/";            
                
                // Note: As activity->pts is an unordered map, need machine name to access
                float t = activity->pts[option->descr]->control[0];
                float percent = 0.05f;
                
                // 95% confience interval [mu - range, mu + range] where range = 1.96 * stdev                
                float mu = t;
                float range = percent * t;
                float stdev = range / 1.96f;
                activity->pts[option->descr]->distr = Distribution::normal;
                activity->pts[option->descr]->control.clear();
                activity->pts[option->descr]->control.push_back(mu);
                activity->pts[option->descr]->control.push_back(stdev);
                activity->pts[option->descr]->show_details<SimTime>(cout);

                // Uniform option: [t, t + delta] where delta = percent * t
            }
        }
        save_instance(filename);
    }

    /* Record all assignments that can be made */
    void JobShop::define_action_space() {
        int i = 0;
        string id;
        for (auto activity : activities) {
            for(auto &machine: activity->options) {
                std::string id = activity->descr + "_" + machine->descr;                                              
                feasible_assignments.push_back(pair<Activity*, Machine*>(activity, machine));       
                assignment_index[id] = i;
                i++;
            }
        }
    }

    /* Get the integer id (a.k.a., index) of the action */
    int JobShop::get_action_index(Action& action) { 
        return action.get<int>(); 
    }

    /* */
    Action JobShop::get_action(const string& descr) {
        int index = find_action(descr);
        return get_action(index); 
    }

    /* */
    Action JobShop::get_action(int index) {
        Action action;
        action.define(index);
        return action;
    }       

    Action JobShop::unhash_action(const string& code) { return get_action(code);}

    int JobShop::find_action(const string& descr) { return assignment_index[descr]; }

    int JobShop::number_of_actions() { return static_cast<int>(feasible_assignments.size()); }

    int JobShop::hash_assignment(Activity* _activity, Machine* _machine) {        
        return _activity->id * static_cast<int>(machines.size()) + _machine->num;
    }

    /* Describe the action with a string */
    string JobShop::hash_action(Action& action) {
        Assignment assignment = get_assignment(action);
        string hash_action = assignment.first->descr + "_" + assignment.second->descr;
        return hash_action;
        // Optional: string hash_action = to_string(hash_assignment(activity, machine));
    }

    /* Different format to load_instance format */
    void JobShop::save_instance(const string &name) {
        int count = 0, options = 0;
        ofstream out(name);
        out << jobs.size() << "," << machines.size();
        for (auto job : jobs) {
            for (auto activity : job->activities) {
                count++;
                options += (int) activity->options.size();
                out << "\nJ" << job->num << "/O" << activity->num;
                for (int i = 0; i < activity->options.size(); i++) {
                    string name = activity->options[i]->descr;
                    int mach_id = activity->options[i]->num;
                    out << "/M" << mach_id << "/";                                       
                    activity->pts[name]->show_details<SimTime>(out);                    
;               }
            }
        }
        cout << "\n" << count << ", " << options;
    }

    /* Data initialisation */
    void JobShop::create_machines(int m) {
        for (int j = 0; j < m; j++) {
            string name = "M" + to_string(j + 1);
            add_machine(new Machine(name, *this, sim));
        }
    }

    /* Data initialisation */
    void JobShop::create_jobs(int n) {
        for (int i = 0; i < n; i++) {
            string name = "J" + to_string(i + 1);
            Job* _job = new Job(name, i, *this, sim);
            add_job(_job);
        }
    }

    /* Data initialisation - FJSP */
    void JobShop::define_activity(int job_id, int act_id, vector<pair<int, Parameter*> > &cand) {
        string descr = jobs[job_id]->descr + "|" + to_string(act_id + 1);
        Activity* act = new Activity(descr, sim, *jobs[job_id], act_id);
        jobs[job_id]->add_activity(*act);
        for (int i = 0; i < (int) cand.size(); i++) {
            act->add_option(machines[cand[i].first], cand[i].second);
        }
    }

    /* Data initialisation */
    void JobShop::add_machine(Machine* mach) {
        machines.push_back(mach);
        mach->num = static_cast<int>(machines.size()) - 1;
    }

    /* Data initialisation */
    void JobShop::add_job(Job* job) { jobs.push_back(job); }

    /* This procedure is used after "::simulate" is called and uses
    the current recorded start time and machine that is stored within the activity object */
    void JobShop::extract_priority_order(JobShop::PriorityOrder &order) {
        multimap<SimTime, pair<Activity*, Machine*> > map;// Create a multimap for sorting
        for (auto job : jobs) {
            for (auto activity : job->activities) {
                map.insert({ activity->start, pair<Activity*, Machine*> (activity, activity->machine) });
            }
        }
        // Extract to list:
        for (const auto& entry : map) order.push_back(entry.second);
    }

    /* Sequencing - Identify from the priority ordering */
    MachineSequences* JobShop::extract_sequences(PriorityOrder& order) {
        MachineSequences* sequences = new MachineSequences; // Create a new unordered_map
        for(auto machine : machines) (*sequences)[machine->descr] = std::vector<Activity*>(); // Create vectors for each machine
        for (auto const& entry : order) // Iterate through priority ordering
            (*sequences)[entry.second->descr].push_back(entry.first); // Update machine's sequence
        return sequences;
    }

    /* Access */
    int JobShop::find_activity(PriorityOrder& order, Activity* activity) {
        int a = 0;
        while (order[a].first != activity) a++;
        return a;
    }

    /* Logging (Reporting) */
    void JobShop::log_schedule(Schedule& schedule, ostream& os) {
        os << "\nSchedule => ";
        for (Job* job : jobs) {
            os << "\n" << job->descr << ": ";
            for (Activity* activity : job->activities) {
                int m = get<2>(schedule[activity->descr]);
                os << machines[m]->descr;
                os << "(" << get<0>(schedule[activity->descr]);
                os << "," << get<1>(schedule[activity->descr]);
                os << ") ";
            }
        }       
    }

    /* Logging (Reporting) */
    void JobShop::log_gantt(Schedule& schedule, ostream& os) {
        os << "\nGantt chart plotting";
        for (Job* job : jobs) {
            string colour = "(";
            colour += to_string((unif_0_1(gen))) + ",";
            colour += to_string((unif_0_1(gen))) + ",";
            colour += to_string((unif_0_1(gen))) + ")";
            for (Activity* activity : job->activities) {
                SimTime start = get<0>(schedule[activity->descr]);
                SimTime end = get<1>(schedule[activity->descr]);
                int m = get<2>(schedule[activity->descr]);
                SimTime dur = end - start; 
                os << "\n['" << activity->descr;
                os << "'," << start << "," << end << ",";
                os << - m - 1<< "," << colour << "],";
           }
        }
    }

    /* Logging (Reporting) */
    void JobShop::log_ordering(PriorityOrder& order, ostream& os) {
        os << "\nOrder => ";
        os << "\n[";
        for (auto &entry : order) {
            os << entry.first->descr << "[" << entry.second->descr << "]";
            if (entry != order.back()) os << ", ";
        }
        os << "]\n";
    }

    /* Logging (Reporting)  */
    void JobShop::log_sequence(vector<Activity*>& sequence, ostream& os) {
        for (auto &entry : sequence) {
            os << entry->descr;
            if (entry != sequence.back()) os << ", ";
        }
    }

    /* Logging (Reporting) */
    void JobShop::log_sequences(MachineSequences& sequences, ostream& os) {
        os << "\nSequences =>";
        for (auto& pair : sequences) {
            os << "\n" << pair.first << ": [";
            log_sequence(pair.second, os);
            os << "]";
        }
        os << "\n";
    }    

    /* An action is an assignment of an activity to a machine */
    void JobShop::get_priority_order(vector<Action> &actions, JobShop::PriorityOrder &order) {
        for (auto action: actions) {
            order.push_back(get_assignment(action));
        }
    }


    /* */
    float JobShop::evaluate_actions(vector<Action>& actions, bool stoch) {
        PriorityOrder order;
        get_priority_order(actions, order);
        Metrics metrics;
        evaluate(order, metrics, stoch);
        float score = metrics.get_cmax();
        return -score;
    }

    /* Build non-delay schedule from a priority ordering - forward scheduling */
    Schedule* JobShop::build_schedule(PriorityOrder& _order, bool sample) {
        vector<SimTime> free(machines.size(), 0); // Set to zero
        Schedule *schedule = new Schedule;
        Activity *activity, *pred;
        int m;
        SimTime start, pt;
        for (const auto& entry : _order) { // entry == pair<activity, machine>
            activity = entry.first; // Current activity
            pred = activity->pred(); // Pointer to previous activity of the job
            m = entry.second->num; // Index of assigned machine            
            pt = activity->get_pt(entry.second, sample); 

            // Note: pred_end_time = get<1>((*schedule)[pred->descr]);
            start = (pred == nullptr) ? free[m] : fmax(free[m], get<1>((*schedule)[pred->descr]));
            tuple<SimTime, SimTime, int> tup(start, start + pt, m);
            (*schedule)[activity->descr] = tup;
            free[m] = start + pt; // Next free at this time
        }
        return schedule;
    }

    /* Build a non-delay schedule from a priority ordering - backward scheduling */
    Schedule* JobShop::build_schedule_reverse(PriorityOrder& _order, SimTime t, bool sample) {
        vector<SimTime> free(machines.size(), t); 
        Schedule* schedule = new Schedule;
        Activity* activity, *succ;
        int m;
        SimTime end, pt;
        for (auto it = _order.rbegin(); it != _order.rend(); it++) {
            activity = (*it).first; // Current activity
            succ = activity->succ();// Pointer to successor activity of the job
            m = (*it).second->num; // Index of assigned machine
            pt = activity->get_pt((*it).second); // Get pt for assigned machine
            // Note: succ_start_time = get<0>((*schedule)[succ->descr]);
            end = (succ == nullptr) ? free[m] : fmin(free[m], get<0>((*schedule)[succ->descr]));
            (*schedule)[activity->descr] = tuple<SimTime, SimTime, int>(end - pt, end, m);
            free[m] = end - pt;
        }
        return schedule;
    }

    /* Identify the critical path */
    vector<Activity*> JobShop::get_critical_activities(PriorityOrder& _order) {
        vector<Activity*> critical;
        bool is_critical;
        string name;
        Schedule* sch = build_schedule(_order, false);
        Metrics metrics;
        evaluate_schedule(*sch, metrics);
        Schedule* rev_sch = build_schedule_reverse(_order, metrics.makespan, false);
        for (auto it = _order.begin(); it != _order.end(); it++) {
            name = (*it).first->descr; // Name of activity
            is_critical = (get<0>((*sch)[name]) == get<0>((*rev_sch)[name]));
            if (is_critical) critical.push_back((*it).first);
        }
        return critical;
    }

    /* State Extraction - Directly from simulated objects */
    SystemState JobShop::extract_state(SimTime t) {
        SystemState _state = System::extract_state(t); // Extract object and state

        for (Job* job : jobs) {
            // ______________________________________________            
            if(is_complete(job->sim_state))
                _state.add_more_info(job->descr, rounded_string(job->last()->end));                
            else
                _state.add_more_info(job->descr, job->current->descr);                
         
            // _______________________________________________
            // Record activity end time, remaining time, and machine assigned
            for (Activity* activity : job->activities) {
                if (is_complete(activity->sim_state)) {
                    _state.add_more_info(activity->descr, activity->machine->descr + "," + rounded_string(activity->end));                   
                }
                else if (is_pending(activity->sim_state)) {
                    ///_state->add_more_info<string>(activity->descr, "?"); // No need for this code if the pt value is machine dependant
                }
                else if(in_progress(activity->sim_state)) {
                    SimTime rem = activity->get_remain(t);          
                    _state.add_more_info(activity->descr, activity->machine->descr + "," + rounded_string(rem));
                    _state.add_more_info(activity->machine->descr, rounded_string(t + rem));                    
                }              
            }
        }
        return _state;
    }

    /* Check whether the provided name is valid */
    bool JobShop::is_job_name(const string& name) {
        for (int i = 0; i < jobs.size(); i++)
            if (jobs[i]->descr.compare(name) == 0) return true;
        return false;
    }

    /* Check whether the provided name is valid */
    bool JobShop::is_activity_name(const string& name) {
        for (int a = 0; a < activities.size(); a++)
            if (activities[a]->descr.compare(name) == 0) return true;
        return false;
    }

    /* Extract the time that is placed at the start of the hashed state */
    SimTime JobShop::extract_time(const string& code) {
        vector<string> str = split_string(code, '|');
        SimTime time = parse_string<SimTime>(str[0]);
        return time;
    }

    /* FORMAT: "17|2,2,4|1,1,14|3,p|2,3,1|" => J1/O3 @ M3, J2/O2 @M2, J3/O3 pend, J4/O3 @M4
    */
    string JobShop::hash_state(SystemState& state) {

        Activity* current;
        string info;
        char delim = '|';
        string code = std::format("{:.2f}", state.time) + delim;

        for (auto job : jobs)
        {
            auto it = state.records.find(job->descr);
            const StateRecord& record = (*it).second; // The jobs record
          
            if(is_complete(record.state)) code = code + "c" + delim;  
            else
            {
                // Get the current activity
                string name = state.get_more_info(job->descr);
                current = get_activity(name); // Get the current activity
                
                auto it = state.records.find(current->descr);
                const StateRecord& activity_record = (*it).second;
                                
                if (is_pending(activity_record.state)) {
                    code = code + std::format("{:}", current->num + 1) + ",p" + delim;                    
                }
                else if (in_progress(activity_record.state)) {
                    info = state.get_more_info(current->descr); // machine + remaining time 
                    info.erase(0, 1); // Remove redundant character 'M' to reduce overhead
                    code = code + std::format("{:}", current->num + 1) + "," + info + delim;                    
                }
            }           
        }
        return code;
    }

    /* Revise pred and succ activity states */
    void JobShop::set_activity_states(SystemState& state, Job* job, int current) {
        for (int a = 0; a < current; a++)
            state.records[job->activities[a]->descr].state = "COMPLETE";

        for (int a = current + 1; a < job->activities.size(); a++)
            state.records[job->activities[a]->descr].state = "PENDING";
    }

    /*  */
    SystemState JobShop::unhash_state(const string& code) {
        SystemState state; // Create a new object
        state.is_terminal = true; // Assume it is until proven otherwise
      
        ////////////////////////////////////////////////////////////////////////////
        /// Stage 1: Create object records for jobs, activities and machines
        for (Job* job : jobs) {            
            state.add(StateRecord(job, "PENDING"));
            for (Activity* activity : job->activities)
                state.add(StateRecord(activity, "PENDING"));
        }
        for (Machine *machine: machines)        
            state.add(StateRecord(machine, "IDLE"));
        
        ///////////////////////////////////////////////////////////////////////////
        // Stage 2: Revise the details according to what is found in the code
        vector<string> str = split_string(code, '|');
        state.time = parse_string<SimTime>(str[0]);
        int a = -1;
        for (int i = 0; i < (int) jobs.size(); ++i) {
            Job* job = jobs[i];             
            auto& job_record = state.records[job->descr];
            Activity* current_activity = nullptr;
            StateRecord* activity_record = nullptr;
            vector<string> sub = split_string(str[i + 1], ','); // 3 options -> "c", "a,p", "a,m,rem"
            if (sub.size() >= 2) {
                a = std::stoi(sub[0]) - 1; // Index of the current job activity
                current_activity = job->activities[a];
                activity_record = &state.records[current_activity->descr];
                
                state.is_terminal = false;  
                state.add_more_info(job->descr, current_activity->descr); // Record activity in job's state record
            }

            switch (sub.size()) {
            case 1: // 'c' => no current activity; job is complete
                job_record.state = "COMPLETE";
                for(Activity *activity: job->activities)
                    state.records[activity->descr].state = "COMPLETE";         
                break;
            case 2: // "#,p" => current activity is pending                      
                // Current activity already initialized to "PENDING"                          
                set_activity_states(state, job, a);               
                break;
            case 3: // Current activity is in-progress; successors are pending, predecessors must be complete                
                job_record.state = "IN_PROGRESS";
                activity_record->state = "IN_PROGRESS";
                set_activity_states(state, job, a);  // Revise predecessor and successor states  

                int m = stoi(sub[1]) - 1; // Index of the assigned machine
                Machine* machine = machines[m];
                float rem = stof(sub[2]); // Remaining processing time
                
                // Revise machine state                
                auto& machine_record = state.records[machine->descr];
                machine_record.state = "BUSY";
                
                // Record detailed info                               
                state.add_more_info(current_activity->descr, std::format("{},{:.2f}", machine->descr, rem)); 
                state.add_more_info(machine->descr, rounded_string(state.time + rem));
                                                                                         
                break;
            }            
        }
        return state;
    }

    /* Record within the system history object, the assignment that was made.
    The assignment is recorded as an Action */
    void JobShop::record_assignment(SimTime t, Activity* activity, Machine* machine) {
        string id = format("{}_{}", activity->descr, machine->descr); // Convert to string
        int index = find_action(id);
        Action action = get_action(index);
        action.when = t;
        record_action(action);
    }

    void JobShop::log_history(ostream& os) {
        history.log_states(os);
        report_actions(os);
    }

    /* Logging */
    void JobShop::report_actions(ostream& os) {
        os << "ACTIONS: ";
        for (Action& action : history.actions) report_action(os, action);
    }

    /* Logging. The action.info is an integer, but it is not informative 
    so report the assignment explicitly
    */
    void JobShop::report_action(ostream& os, Action& action) {
        Assignment assignment = get_assignment(action);
        os << assignment.first->descr << "@" << assignment.second->descr <<"{" << action.when<<"} ";
    }

    /* Reinf. Learning Method: Identify assignments that can be made from the current state 
    using the object states
    ASSUMPTION: Every object must have a record in state.records
    */
    void JobShop::get_available_actions(SystemState& state, Actions& actions) {
        Machine* mach;
        Activity* act, *pred_act;
        Assignment assignment;
        actions.clear(); // Clear previous

        /// Iterate through each of the stored possibilities. 
        /// An assignmant is feasible if, i) the machine is idle;
        /// ii) the activity is pending, and iii) the predecessor is complete
        for (int a = 0; a < (int) feasible_assignments.size(); a++) {
            assignment = feasible_assignments[a];
            act = assignment.first;
            const StateRecord* ref = state.lookup_record(act->descr); 
           
            if(is_pending(ref->state)) {
                pred_act = act->pred(); // The predecessor activity
                const StateRecord* activity_record = nullptr;
                if (pred_act != nullptr) 
                    activity_record = state.lookup_record(pred_act->descr);

                if (activity_record == nullptr or is_complete(activity_record->state)) {
                    mach = assignment.second;
                    const StateRecord* machine_record = state.lookup_record(mach->descr);
                    if(is_idle(machine_record->state)) { 
                        Action action;
                        action.define(static_cast<int>(a));
                        actions.add(action); 
                    }
                }
            }
        }
    }

    /* Reinf. Learning Method: Check whether futher actions are possible */
    bool JobShop::is_terminal_state(SystemState& state) {
        Job* job;
        for (const auto& entry : state.records) {
            const StateRecord& record = entry.second;          
            if (is_job(record.obj, job)) {
                if(not_complete(record.state)) return false;
            }
        }
        return true;
    }

    Assignment JobShop::get_assignment(Action& action) {
        int a = action.get<int>();
        return feasible_assignments[a];
    }

    /* Reinf. Learning Method: Assign an activity to a machine and change related states.
    The timing of the new state is tied directly to the "completion" of the
    action and the resulting system change.
    Assumption: SimTime is formatted to 2DP
    */
    void JobShop::apply_action(SystemState& state, Action &action) {

        /// Step 1: Extract the assignment
        Assignment assignment = get_assignment(action); 

        /// Step 2: Define shortcuts
        Activity* activity = assignment.first; 
        Machine* machine = assignment.second; 
        SimTime pt = activity->get_pt(machine); 

        /// Step 3: Change the state
        state.records[activity->descr].state = "IN_PROGRESS";
        state.add_more_info(activity->descr, machine->descr + "," + rounded_string(pt));        
        
        state.records[machine->descr].state = "BUSY";
        state.add_more_info(machine->descr, rounded_string(state.time + pt));              
    }

    /*
        Compute the partial/projected makespan based upon in-progress activities
        Assumption: Completed activities cannot affect the future.
    */
    float JobShop::compute_projected_cmax(SystemState& state) {
        float _cmax = state.time;
        for (auto activity : activities) {
            if (state.records.find(activity->descr) != state.records.end()) {
                const StateRecord& record = state.records[activity->descr];
                if (in_progress(record.state)) {
                    string info = state.get_more_info(activity->descr);
                    vector<string> str = split_string(info, ',');
                    SimTime rem_pt = parse_string<SimTime>(str[1]);
                    SimTime end = state.time + rem_pt;
                    _cmax = max(_cmax, end);
                }
            }
        }
        return _cmax;
    }

    float JobShop::get_completed_jobs(SystemState& state) {
        int n = 0, count = 0;
        for (auto job : jobs) {
            const StateRecord& record = state.records[job->descr];
            if (is_complete(record.state)) count++;
            n++;
        }       
        return float(count) / float(n);
    }
    
    float JobShop::get_completed_operations(SystemState& state) {
        int n = 0, count = 0;
        for (auto activity : activities) {
            const StateRecord& record = state.records[activity->descr];
            if (is_complete(record.state)) count++;
            n++;
        }
        return float(count) / float(n);
    }

    
    float JobShop::compute_reward(SystemState& previous_state, SystemState& next_state, Action action)
    {
        switch (reward) {
        case RewardType::incr: return reward_function_incr(previous_state, next_state, action);              
        case RewardType::delayed: return reward_function_delay(previous_state, next_state, action);
        default: throw std::invalid_argument("Unknown reward function");
        }
    }

    /* Define reward as incremental change in makespan */
    float JobShop::reward_function_incr(SystemState& previous_state, SystemState& next_state, Action action) {
        float cmax_prev = compute_projected_cmax(previous_state);
        float cmax_next = compute_projected_cmax(next_state);
        float diff = cmax_next - cmax_prev; // Always positive
        
        /// Reward shaping:
        //float bonus = get_completed_jobs(next_state);
        //float bonus = get_completed_operations(next_state);
        return -(cmax_next - cmax_prev); // +0.01 * bonus;  // weight in [0.01,0.1, 1, 10, 100]
        
        // TODO: Add bonus for fraction of completed jobs or operations at next_state

        // OR return (1.0 / diff);
        // float ratio;
        // int n = compute_number_completed(next_state, ratio);        
    }

    /* Define reward as the incumbent makespan */
    float JobShop::reward_function_incum(SystemState& previous_state, SystemState& next_state, Action action) {
        float neg_cmax = - compute_projected_cmax(next_state);
        return (float) neg_cmax; // + 10*ratio
        
        // OR return (float) (1.0 / cmax_next);
        // float ratio;
        // int n = compute_number_completed(next_state, ratio);        
    }

    /* Define reward as zero or as the makespan */
    float JobShop::reward_function_delay(SystemState& previous_state, SystemState& next_state, Action action) {        
        if (next_state.is_terminal) 
            return - compute_projected_cmax(next_state);
        else 
            return 0;
    }
        
    /* Direct approach to identify the next decision point - from the specified state 
    Assumption: "state" is the current state - if not need to instate it */
    optional<SimTime> JobShop::get_next_decision_point(SystemState& state, bool exclude_current) {
        set<SimTime, std::less<SimTime> > times; 
        string info;
        for (auto _machine : machines) {

            const StateRecord *record = state.lookup_record(_machine->descr);
            
            /// Machine is available at a future time
            if(is_busy(record->state)) {
                info = state.get_more_info(_machine->descr);
                SimTime time = parse_string<SimTime>(info);                
                times.insert(time);
            }

            /// Machine is available now
            if(is_idle(record->state)) {
                if (!exclude_current) times.insert(state.time);                    
            }
        }
       
        if (!times.empty()) return (*times.begin()); // Grab the earliest time     
        return nullopt; // Terminal state reached
    }

   /* Transition system to next time point */
    void JobShop::transition_to_time(SystemState& state, SimTime next_point) {              
        for (auto &activity : activities) {

            auto& activity_record = state.records[activity->descr];

            if (!in_progress(activity_record.state))
                continue; // Bug out - do not continue

            string activity_info = state.get_more_info(activity->descr); // String "machine,rem_pt"
            
            vector<string> str = split_string(activity_info, ',');
            string assigned = str[0]; // The assigned machine
            SimTime rem_pt = parse_string<SimTime>(str[1]); // Convert string to SimTime                                        
            SimTime end_time = state.time + rem_pt; // The activity completion time

            if (end_time <= next_point) { // Completed before the next time point
                activity_record.state = "COMPLETE";                
                state.add_more_info(activity->descr, std::format("{},{:.2f}", assigned, end_time));
                state.records[assigned].state = "IDLE"; // Revise machine state

                if (auto * next = activity->succ()) 
                    state.add_more_info(activity->job.descr, next->descr);
                else 
                    state.records[activity->job.descr].state = "COMPLETE";                                    
            }
            else { // Still in progress
                rem_pt = end_time - next_point; // Revise remaining processing time info
                state.add_more_info(activity->descr, std::format("{},{:.2f}", assigned, rem_pt)); // i.e., assigned + "," + rounded_string(rem_pt));              
            }
        }
        state.time = next_point;
    }

    void JobShop::restore_machine(Machine* machine, const StateRecord& record, SystemState& state) {
        machine->sim_state = record.state;
    }

    void JobShop::restore_pending_activity(Activity* activity, SystemState& state) {
        activity->sim_state = "PENDING";
        activity->machine = nullptr; // No assignment
        activity->start = activity->end = -1; // No value yet
        
         Activity* pred = activity->pred(); // Earlier job activity

        // If predecessor activity exists, initialise with the recorded state         
        if (pred != nullptr) {
            auto it = state.records.find(pred->descr);
            pred->sim_state = it->second.state; 
        }
       
        // Enqueue the activity
        if (pred == nullptr || is_complete(pred->sim_state)) {
            activity->job.set_current(activity); // Tell the related job what the current activity is            
            enqueue_activity(activity); // Automatically add to the queue

            /// Optional Approach - Schedule the first activity
            //if (pred == nullptr) schedule_first_activity(&activity->job);
            //else enqueue_activity(activity);
        }
    }

    /* An in-progress activity has a remaining time and a machine assigned information */
    void JobShop::restore_in_progress_activity(Activity* activity, SystemState& state) {
        activity->job.current = activity; // Set as current job activity        
        activity->sim_state = "IN_PROGRESS";
        if (activity->num == 0) activity->job.sim_state = "IN_PROGRESS"; 

        // Add addiotnal information
        string info = state.get_more_info(activity->descr);
        vector<string> sub = split_string(info, ',');
        activity->assign_machine(get_machine(sub[0]));
        SimTime rem = parse_string<SimTime>(sub[1]);
        
        // Schedule future completion event
        sim.schedule_event(state.time + rem, "COMPLETION", activity, manager);
    }

    /* Completed activities no longer affect the future state. Just record completion status */
    void JobShop::restore_completed_activity(Activity* activity, SystemState& state) {
        activity->sim_state = "COMPLETE"; 
        if (!activity->succ()) activity->job.sim_state = "COMPLETE";
    }

    /* To run the simulator from the current state 
       need to update states and attributes of jobs, activities and machine
    */
    void JobShop::restore_state(SystemState& state) {
        sim.set_now(state.time); // Reset simulation clock 
        sim.delete_event_queue(); // Delete scheduled events
        history.clear(); // Clear all stored states
        clear_queue(); // Empty the ready_to_process queue

        for (const auto& [_, record] : state.records) {
            if (Machine* machine = as_machine(record.obj)) {
                restore_machine(machine, record, state);
            }
            else if (Job* job = as_job(record.obj)) {
                job->sim_state = record.state;
            }
            else if (Activity* activity = as_activity(record.obj)) {        
                if (is_pending(record.state)) 
                    restore_pending_activity(activity, state);
                else if (in_progress(record.state)) 
                    restore_in_progress_activity(activity, state);
                else if (is_complete(record.state)) 
                    restore_completed_activity(activity, state);
            }
        }  
    }


    /* Access Function. name = J# => str = ["", "#"] */
    Job* JobShop::get_job(const string& name) {
        vector<string> str = split_string(name, 'J');
        int j = stoi(str[1]) - 1;
        return jobs[j];
    }
    
    /* Access Function. name = J#|# */
    Activity* JobShop::get_activity(const string& name) {
        vector<string> str = split_string(name, 'J'); 
        str = split_string(str[1], '|');
        int j = stoi(str[0]) - 1;
        int a = stoi(str[1]) - 1;
        return jobs[j]->activities[a];
    }

    /* Access Function. name = M# */
    Machine* JobShop::get_machine(const string& name) {
        vector<string> str = split_string(name, 'M'); // str = ["", #]
        int index = std::stoi(str[1]) - 1;
        return machines[index];
    }

    /* Within the priority ordering, an activity should be postioned earlier if its start time is earlier */
    void JobShop::fix_ordering(PriorityOrder& order) {
        order.clear();
        extract_priority_order(order); // Define from the current schedule
    }

    /* Create a list of the times when activities begin*/
    list<SimTime>* JobShop::extract_time_points() {
        map<SimTime, Activity*> _map;
        for (auto job : jobs) {
            for (auto activity : job->activities) {
                _map.insert({ activity->start, activity }); // Chronologically ordered
            }
        }
        // Extract the list from the map
        list<SimTime>* times = new list<SimTime>{0};         
        for (const auto& entry : _map) {
            if (entry.first != times->back()) times->push_back(entry.first);
        }
        return times;
    }

    /* */
    void JobShop::copy_order(PriorityOrder& order, PriorityOrder& copy) {
        copy.clear();
        for (auto& pair : order) {
            Assignment copy_pair(pair.first, pair.second);
            copy.push_back(copy_pair);
        }
    }


    /*********************************************************/
    /* --------------------- SA CLASS ---------------------- */
    /*********************************************************/

    Solution::Solution(JobShop::PriorityOrder& _order, Metrics&_metrics):
        order(_order), metrics(_metrics) {}

    Solution::~Solution() {}

    ISa::ISa(int _id, std::mt19937& _gen, JobShop& _shop, Solution *starting, string outfile, bool _stoch)
        :id(_id), gen(_gen), shop(_shop), current(starting), stoch(_stoch), best(nullptr)
    {
        par_eval = true;
        rate = (float) exp( (1.0 / (temp_steps - 1)) * log(tF / tI)); // Geometric decrease
        //rate = (tI - tF) / (temp_steps); // Optional: Linear decrease

        out.open(outfile);
        log_solution(current, out, false);                
    }

    ISa::~ISa() {
        delete best;
        delete current;
    }

    void ISa::copy_solution(Solution &solution, Solution &copy) {
        shop.copy_order(solution.order, copy.order);
        copy.metrics.makespan = solution.metrics.makespan;
        copy.metrics.flowtime = solution.metrics.flowtime;
        copy.metrics.nb_scen = solution.metrics.nb_scen;
    }

    void ISa::prepare() {
        t = tI;
        step = no_impr = 0;
        JobShop::PriorityOrder copy_order;
        shop.copy_order(current->order, copy_order);
        Metrics metric;
        metric.makespan = current->metrics.makespan;
        metric.flowtime = current->metrics.flowtime;
        metric.nb_scen = current->metrics.nb_scen;
        best = new Solution(copy_order, metric);
    }

    bool ISa::accept_change(float before, float after) {
        float rv = (float) unif_0_1(gen);
        
        // Always accept if after <= before
        float rhs = exp((before - after)/t);        
        bool accept = (rv <= rhs);
        return accept;
        
        // Optional approach:
        //float lhs = (float) std::log(rv);       
        //float rhs = (before - after) / t;
        //bool accept = (lhs <= rhs);
    }

    void ISa::log_solution(Solution *sol, ostream& os, bool sample) {

        os << "\nScore =>" << sol->metrics.makespan << "(" << sol->metrics.nb_scen << ")";
        os << "\nCPU => " << runtime << "-> " << wall_time <<"\n";

        shop.log_ordering(sol->order, os);
         
        MachineSequences* sequences = shop.extract_sequences(sol->order);
        shop.log_sequences(*sequences, os);

        Schedule* schedule = shop.build_schedule(sol->order, sample);
        shop.log_schedule(*schedule, os);
        os << "\n";

        shop.log_gantt(*schedule, os);
        os << "\n";

        delete schedule;
        delete sequences;
    }

    void ISa::restore_best() {
        copy_solution(*best, *current);
        evaluate(current);     
    }

    /* Local Search */
    void ISa::local_search(Solution *solution, bool all) {
        if (all) {
            for (int a = 0; a < static_cast<int>(solution->order.size()); a++) {
                local_search_shift(solution, a);
                local_search_machine(solution, a);
            }
        }
        else {            
            local_search_shift(solution);            
            local_search_machine(solution);
        }

        // Optional: if(t > 0.9*tF) local_search_exchange(solution);
    }

    /* Try all possible shifts */
    void ISa::local_search_shift(Solution* solution, int a) {
        SimTime score = solution->metrics.makespan; // Reference to original score
        JobShop::PriorityOrder& order = solution->order; // Shortcut
        int n = (int) order.size(); // Shortcut
        Activity* activity = order[a].first; // Choose an activity
        int left, right; // Extent of the shift permitted
        for (left = a; left > 0 and &order[left - 1].first->job != &activity->job; --left); // Find left boundary
        for (right = a; right < n - 1 and &order[right + 1].first->job != &activity->job; ++right); // Find right boundary
        if (right - left > 1) {
            int best = a;
            for (int a_ = left; a_ <= right; a_++) {
                if (a_ == a) continue; // Skip to next value of a_                                
                if (order[a_].second != order[a].second) continue; // Skip - Must be same machine, otherwise no change in sequence
                
                shift(a, a_, order); // Perturb the ordering
                SimTime _score = evaluate(solution); // Evaluate perturbation
                shift(a_, a, order); // Under perturbation
                if (_score < score) { best = a_;  score = _score; }
            }
            solution->metrics.makespan = score; // Reset the score
            if (a != best) { // There is a shift that can improve the solution                
                shift(a, best, order); // Refine solution               
                update_best(solution);
            }
            //evaluate(solution); // Optional: Reset activity timings at extra computational burden.
        }
    }

    /* Local Search */
    void ISa::local_search_shift(Solution* solution) {
        JobShop::PriorityOrder& order = solution->order; // Shortcut
        std::uniform_int_distribution<> distr(0, static_cast<int>(order.size()) - 1); // Define distribution
        local_search_shift(solution, distr(shop.gen));// Choose an activity within the ordering        
    }

    /* Try all shifts of one position - Only useful in early search */
    void ISa::local_search_exchange(Solution* solution) {
        SimTime score_before = solution->metrics.makespan; // Reference to original score
        JobShop::PriorityOrder& order = solution->order; // Shortcut        
        for (int i = 1; i < static_cast<int>(order.size()); i++) {
            shift(i, i-1, order); // Perturb the ordering
            SimTime score_after = evaluate(solution); // Evaluate perturbation
            if (score_after < score_before) { update_best(solution); } // Optional: if (accept_change(score_before, score_after))                            
            else {
                shift(i - 1, i, order); // Undo the perturbation
                solution->metrics.makespan = score_before; // Reset score
            }
        }
        update_best(solution);
    }

    /* Local Search */
    void ISa::local_search_machine(Solution* solution, int a) {
        JobShop::PriorityOrder& order = solution->order; // Shortcut
        SimTime score = solution->metrics.makespan; // Reference to original score
        Activity* activity = order[a].first; // Choose an activity
        Machine* current_machine = order[a].second; // Reference to current assigned machine
        Machine* best = order[a].second;
        for (auto machine : activity->options) {
            if (machine != current_machine) {
                order[a].second = machine;
                SimTime _score = evaluate(solution); // Evaluate the perturbation
                if (_score < score) { score = _score; best = machine; }
            }
        }

        // Return original assignment or make change
        order[a].second = best;
        solution->metrics.makespan = score;
        update_best(solution);

        //evaluate(solution); // Optional: Reset activity timings at extra computational burden.
    }

    /* Local Search */
    void ISa::local_search_machine(Solution* solution) {
        JobShop::PriorityOrder& order = solution->order; // Shortcut
        vector<int> cand;
        int a = 0;
        for (auto &entry: order) {
            if (entry.first->options.size() > 1) cand.push_back(a);
            a++;
        }
        std::uniform_int_distribution<> distr(0, static_cast<int>(cand.size()) - 1); // Define distribution 
        local_search_machine(solution, distr(shop.gen));// Choose an activity within the ordering
     }

    /* Perturbation */
    void ISa::perturb_solution(Solution* solution) {
        float val = (float) unif_0_1(shop.gen);
        //if(val < 0.33)
        //perturb_assignment(solution, false);
        //else if (val < 0.66)
        perturb_shift(solution, false);
        //else 
        //perturb_block(solution);       
        
    }

    /* Perturbation */
    void ISa::shift(int a, int b, JobShop::PriorityOrder& order) {
        if(a == b) throw std::invalid_argument("Error a cannot be equal to b");        
        pair<Activity*,Machine*> temp = order[a];
        if (a < b) // Shift activities one position to the left
        { for (int i = a; i < b; ++i) order[i] = order[i + 1]; }
        else // Shift activities one position to the right
        { for (int i = a; i > b; --i)  order[i] = order[i - 1]; }
        order[b] = temp; // Copy to re-insertion point
    }

    /* Perturbation */
    void ISa::perturb_shift(Solution* solution, bool crit) { 
        SimTime score_before = solution->metrics.makespan; // Reference to original score
        JobShop::PriorityOrder& order = solution->order; // Shortcut
        int n, a_, a;
        Activity* activity = nullptr;

        /*if (crit) {
            vector<Activity*> critical = shop.get_critical_activities(order);
            n = critical.size();
            std::uniform_int_distribution<> distr(0, n - 1); // Define distribution
            int choose = distr(generator); // Choose an activity within the ordering
            activity = critical[choose]; // The activity to move
            a = shop.find_activity(order, activity);
        }
        else {*/
            n = static_cast<int>(order.size());
            std::uniform_int_distribution<> distr(0, n - 1); // Define distribution
            a = distr(shop.gen); // Choose an activity within the ordering
            activity = order[a].first; // The activity to move
        //}

        int left, right; // Extent of the shift permitted
        for (left = a; left > 0 and &order[left - 1].first->job != &activity->job; --left); // Find left boundary
        for (right = a; right < n - 1 and &order[right + 1].first->job != &activity->job; ++right); // Find right boundary
        if (right - left > 1) {
            std::uniform_int_distribution<> where(left, right); // Define distribution            
            do { a_ = where(shop.gen); } while (a_ == a); // Choose location                       
            shift(a, a_, order); // Perturb the ordering
            //local_search_machine(solution, a); // NEW FEATURE
            SimTime score_after = evaluate(solution); // Evaluate perturbation               
            if (accept_change(score_before, score_after)) {                 
                accept++; 
                //local_search_machine(solution, a); // NEW FEATURE
                update_best(solution);
            }
            else {
                shift(a_, a, order); // Undo the perturbation
                solution->metrics.makespan = score_before; // Reset the score                
            }            
        }
    }

    /* This version only includes ? of the same machine*/
    void ISa::perturb_shift_advanced(Solution* solution, bool crit) {
        SimTime score_before = solution->metrics.makespan; // Reference to original score
        JobShop::PriorityOrder& order = solution->order; // Shortcut
        int n, a;
        Activity* activity;
        Machine* machine;

        if (crit) {
            vector<Activity*> critical = shop.get_critical_activities(order);
            n = static_cast<int>(critical.size());
            std::uniform_int_distribution<> distr(0, n - 1); // Define distribution
            int choose = distr(shop.gen); // Choose an activity within the ordering
            activity = critical[choose]; // The activity to move
            a = shop.find_activity(order, activity);
            machine = order[a].second; // The assigned machine
        }
        else {
            n = static_cast<int>(order.size()); // Shortcut
            std::uniform_int_distribution<> distr(0, n - 1); // Define distribution
            a = distr(shop.gen); // Choose an activity within the ordering
            activity = order[a].first; // The activity to move
            machine = order[a].second; // The assigned machine
        }

        pair<Activity*, Machine*> *ptr;
        vector<int> cand;

        int right = a+1;
        while (right < n) {
            ptr = &order[right];
            if (&ptr->first->job == &activity->job) break;
            if (ptr->second == machine) cand.push_back(right);
            right++;
        }

        int left = a - 1;
        while (left >= 0) {
            ptr = &order[left];
            if (&ptr->first->job == &activity->job) break;
            if (ptr->second == machine) cand.push_back(left);
            left--;
        }

        if (cand.size() > 0) {
            std::uniform_int_distribution<> choose(0, static_cast<int>(cand.size()) - 1); // Define distribution
            int a_ = cand[choose(shop.gen)]; // Choose where to shift to
            shift(a, a_, order); // Perturb the ordering
            SimTime score_after = evaluate(solution); // Evaluate perturbation
            if (accept_change(score_before, score_after)) { 
                    accept++;  
                    update_best(solution); 
                    local_search_machine(solution, a); // NEW FEATURE
            }
            else {
                shift(a_, a, order); // Undo the perturbation
                solution->metrics.makespan = score_before; // Reset the score
            }
        }
    }

    /* Perturbation */
    void ISa::perturb_assignment(Solution* solution, bool crit) {
        SimTime score_before = solution->metrics.makespan; // Reference to original score
        JobShop::PriorityOrder& order = solution->order; // Shortcut
        vector<int> cand;
        int a = -1;
        
        /*if (crit) {
            /*vector<Activity*> critical = shop.get_critical_activities(order);
            int index = 0;
            for (const auto& entry: critical) {
                if (entry->options.size() > 1) cand.push_back(index); // Include only if there are options
                index++;
            }
            
            int n = (int) cand.size();
            std::uniform_int_distribution<> distr(0, n - 1); // Define distribution
            int choose = distr(generator); // Choose an activity within the ordering
            activity = critical[choose]; // The activity to move
            a = shop.find_activity(order, activity);
        }
        //else {*/
        int index = 0;
        for (const auto& entry : order) {
            if (entry.first->options.size() > 1) cand.push_back(index); // Include only if there are options
            index++;
        }
        std::uniform_int_distribution<> distr(0, static_cast<int>(cand.size()) - 1); // Define distribution
        a = distr(shop.gen); // Choose a candidate activity
        Activity *activity = order[a].first; // Reference to chosen activity
        //}

        if (a < 0) throw std::runtime_error("Error in perturb_assignment");

        // ___________________________________________________
        Machine* temp = activity->machine; // Reference to assigned machine
        vector<Machine*> options;
        for (auto machine : activity->options) {
            if (machine != temp) options.push_back(machine); // Don't include current machine assigned
        }
        if (options.size() >= 1) {
            std::uniform_int_distribution<> distr_2(0, static_cast<int>(options.size()) - 1);
            order[a].second = options[distr_2(shop.gen)]; // Assign another machine
            //local_search_shift(solution, a); // NEW FEATURE
            SimTime score_after = evaluate(solution); // Evaluate the perturbation           
            if (accept_change(score_before, score_after)) { 
                accept++; 
                local_search_shift(solution, a); // NEW FEATURE
                update_best(solution);
            }
            else {
                order[a].second = temp; // Undo the perturbation
                solution->metrics.makespan = score_before; // Reset the score
            }
        }
    }


    /* Evaluation */
    SimTime ISa::evaluate(Solution* solution) {
        shop.evaluate(solution->order, solution->metrics, stoch);
        return solution->metrics.makespan;
    }

    /* */
    void ISa::update_best(Solution* solution) {
        if (solution->metrics.makespan < best->metrics.makespan) {
            copy_solution(*solution, *best);
            improved_best = true; 
        }
    }

    /* */
    void ISa::handle_non_improvement() {
        no_impr = (improved_best) ? 0 : no_impr + 1;
        if (no_impr > max_no_impr) { restore_best(); no_impr = 0; }
    }

    /* */
    void ISa::mutate_solution(Solution* solution) {        
        JobShop::PriorityOrder& order = solution->order; // Shortcut      
        int n = static_cast<int>(order.size());
        std::uniform_int_distribution<> distr(0, n - 1); // Define distribution
        int a_, a = distr(gen); // Choose an activity within the ordering
        Activity *activity = order[a].first; // The activity to move
        int left, right; // Extent of the shift permitted
        for (left = a; left > 0 and &order[left - 1].first->job != &activity->job; --left); // Find left boundary
        for (right = a; right < n - 1 and &order[right + 1].first->job != &activity->job; ++right); // Find right boundary
        if (right - left > 1) {
            std::uniform_int_distribution<> where(left, right); // Define distribution            
            do { a_ = where(gen); } while (a_ == a); // Choose location                       
            shift(a, a_, order); // Perturb the ordering
            SimTime score_after = evaluate(solution); // Evaluate perturbation                           
            update_best(solution);
        }
    }

    /* inner loop */
    void ISa::evaluate_state_changes() {
        improved_best = false; // Altered in ISa::update_best  
        accept = 0;       
               
        for (int count = 0; count < max_iter; count++) {
            perturb_solution(current);            
            //local_search(current, false); // Optional
        } 
        //mutate_solution(current);// Optional: Apply a single mutation

        local_search(best, true); // Optional
        handle_non_improvement();        
        revise_temperature_geometric();     
    }

    void ISa::report_progress() { 
        clock_t now = clock();
        int elapsed =  (int)((now - start) / CLOCKS_PER_SEC);
        cout << "\n[" << id << "] elapsed="<< elapsed<< "; step = " << step<< "; temp = " << t << "; current = " << current->metrics.makespan;
        cout << "; best = " << best->metrics.makespan << ";";
    }

    bool ISa::frozen() { return (t <= tF); }

    void ISa::finalise() { log_solution(best, out, false); }

    void ISa::iterate() {
        terminate = frozen();
        if (terminate) 
            restore_best(); 
        else 
            evaluate_state_changes();        
    }

    /* main algorithm - outer loop */
    void ISa::run() {
        prepare();
        start = clock();
        while (true) {
            iterate();
            report_progress();
            if(terminate) break;
        }
        end = clock(); 
        runtime = (int)((end - start) / CLOCKS_PER_SEC);  

        wall_end = omp_get_wtime();
        wall_time = wall_end - wall_start;

        finalise();
    }


    /*********************************************************/
    /* -------------------- PARSA CLASS -------------------- */
    /*********************************************************/

    /* Constructor */
    PISA::PISA(JobShop& _shop, int _nbsa, bool _stoch, bool _par_eval)
        :shop(_shop), nbsa(_nbsa), stoch(_stoch), par_eval(_par_eval)
    {
        start = -1;
        end = -1;
        runtime = -1;
        best_at = -1;
        
        for (int i = 0; i < nbsa; i++) {
            shop.restore_state(*shop.state_zero);
            SimResult result_i;
            shop.simulate(result_i); // Generate a starting solution 

            ofstream out("history_" + to_string(i) + ".txt");
            shop.log_history(out);

            JobShop::PriorityOrder order;
            shop.extract_priority_order(order);
            Metrics metrics;
            shop.evaluate(order, metrics, stoch);
            Solution* solution = new Solution(order, metrics);
            string outfile = "sa" + to_string(i) + ".res";
            ISa* isa = new ISa(i, shop.gen, shop, solution, outfile, stoch);
            isa->par_eval = par_eval;
            solvers.push_back(isa);
            cout << "\nSA-"<<i << " / cmax=" << metrics.makespan;
        }
    }

    /* */
    PISA::~PISA() {
        cout << "\nDestroying PISA";
        clear();
    }
    
    /* */
    void PISA::clear() {
        for (auto solver : solvers) delete solver;
         solvers.clear();
    }

    /* */
    void PISA::log_solution(string file, bool sample) {
        ofstream os(file);

        solvers[best_at]->log_solution(solvers[best_at]->best, os, sample);
    }

    /* */
    int PISA::get_best() {
        int best_at = 0;
        SimTime best_score = solvers[0]->best->metrics.makespan;
        for (int i = 1; i < nbsa; i++)
            best_at = (solvers[i]->best->metrics.makespan < best_score) ? i : best_at;
        return best_at;
    }

    /* */
    void PISA::share_best() {
        best_at = get_best();
        for (int i = 0; i < nbsa; i++) { 
            if(i != best_at) solvers[i]->update_best(solvers[best_at]->best);
        }
    }

    /* Reporting */
    void PISA::report_progress() { 
        for (auto& solver : solvers) solver->report_progress(); 
    }

    /* */
    bool PISA::has_converged() {
        bool cont = true;
        for (auto& solver : solvers) cont = cont and !solver->terminate; 
        return cont;
    }
    
    /* */
    void PISA::run() {         
        omp_set_dynamic(0); // Explicitly disable dynamic teams

        for (auto& solver : solvers) {
            solver->prepare(); 
            solver->start = clock();
        }

        start = clock();
        //#pragma omp parallel num_threads(nbsa)
        {
            for (int step = 0; step < solvers[0]->temp_steps; step++) {                               

                #pragma omp for 
                for(int i = 0; i < solvers.size(); i++) solvers[i]->iterate();                

                #pragma omp barrier
 
                #pragma omp master 
                {
                    if (step % 1 == 0) share_best();
                    report_progress();
                }
            }
        }

        end = clock();
        runtime = (int)((end - start) / CLOCKS_PER_SEC);
        
        wall_end = omp_get_wtime();
        wall_time = wall_end - wall_start;
        
        share_best();

        for (auto& solver : solvers) {
            solver->end = clock();  
            solver->runtime = (int)((end - start) / CLOCKS_PER_SEC);
            solver->wall_time = wall_end - wall_start;
            solver->finalise();
        }
    }
}

