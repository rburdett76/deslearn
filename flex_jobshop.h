#pragma once

/*********************************************************/
/* -------------------- LIBRARIES  --------------------- */
/*********************************************************/

#include <map>
#include <omp.h>        // OpenMP parallel processing
#include <unordered_set>
#include <thread>
#include <chrono>

#include "simulator.h"
#include "parameter.h"


namespace FlexJobShop {
     
    // NOTE: EventTypes = {UND, ARRIVAL, COMPLETION, COMMENCEMENT}
    // NOTE: ObjStates = {UND, PENDING, IN_PROGRESS, COMPLETE, IDLE, BUSY}

    /*********************************************************/
    /* ----------------- PRE-DEFINITIONS ------------------- */
    /*********************************************************/

    class Activity;
    class Job;
    class Machine;
    class Dispatcher;
    class JobShop;

    /*********************************************************/
    /* --------------------- FUNCTIONS --------------------- */
    /*********************************************************/
 
    SimTime choose_pt(int type);
    vector<Activity*> extract_queue(queue<Activity*> queue, std::mt19937* gen);

    bool is_machine(SimObject* obj, Machine& machines);
    bool is_job(SimObject* obj, Job& job);
    bool is_activity(SimObject* obj, Activity& activity);
    
    bool is_complete(ObjState _state);
    bool not_complete(ObjState _state);
    bool is_pending(ObjState _state);
    bool in_progress(ObjState _state);
    bool is_idle(ObjState _state);
    bool is_busy(ObjState _state);

    /*********************************************************/
    /* --------------------- TEMPLATE ---------------------- */
    /*********************************************************/

    /* Convert string to type T */
    template <class T>
    T parse_string(string time_str) {
        if constexpr (std::is_same_v<T, int>)
            return std::stoi(time_str);
        else if constexpr (std::is_same_v<T, float>)
            return std::stof(time_str);
        else if constexpr (std::is_same_v<T, float>)
            return round2dp(std::stod(time_str));
        else throw;
    }

    /*********************************************************/
    /* ------------------ RESULTS CLASS -------------------- */
    /*********************************************************/

    class Metrics {
    public:
        /* Attribute */
        SimTime makespan; // expected
        SimTime flowtime; // expected
        SimTime stdev_makespan, stdev_flowtime;
        int nb_scen = 1;
        SimTime (Metrics::*score)();  // Function pointer

        /* Function */
        Metrics();
        ~Metrics() {}
        void copy(Metrics* metrics) { makespan = metrics->makespan; flowtime = metrics->flowtime; }
        SimTime get_cmax() { return makespan;  } // KPI 1
        SimTime get_ftime() { return flowtime;  } // KPI 2
    };
    
    /*********************************************************/
    /* ----------------- DERIVED CLASS --------------------- */
    /*********************************************************/

    class Job : public SimObject {
    public:
        /* Attribute */
        int num;
        string descr;
        JobShop& shop;
        vector<Activity*> activities;
        SimTime rlt; // Release time
        Activity* current; // Current activity to be processed

        /* Function */
        Job(string _descr, int _num, JobShop& _shop, Sim& _sim);
        ~Job();
        void add_activity(Activity& _activity);
        void log_details(ostream& os);
        void set_current(Activity* activity);
        Activity* first();
        Activity* last();
        Activity* succ(Activity* activity);
        Activity* pred(Activity* activity);
        SimCoroutine run() override;
        void copy(Job &source);
    };

    /*********************************************************/
    /* ---------------------- CLASS ------------------------ */
    /*********************************************************/

    class Activity : public SimObject {
    public:      
        /* Attribute */
        int id; // Unique id
        int num;// Index within job
        Job& job;
        SimTime start, end;
        Parameter* selected_pt;
        vector<Machine*> options;
        std::unordered_map<std::string, Parameter*> pts; // Fixed or stochastic processing times 
       
        Machine* machine;

        /* Function */
        Activity(string _descr, Sim& _sim, Job& _job, int _num); 
        ~Activity();
        void add_option(Machine* _mach, Parameter* pt);
        void log_details(ostream& os);
        void copy(Activity &source);

        SimTime get_remain(SimTime t);
        void assign_machine(Machine* mach);
        SimTime get_pt(Machine* _machine, bool sample=true);
        Activity* succ();
        Activity* pred();
        SimCoroutine run() override;
    };

    /*********************************************************/
    /* ----------------- DERIVED CLASS --------------------- */
    /*********************************************************/

    class Machine : public SimObject {
    public:
        /* Attribute */
        string descr;
        int num; // Index within JobShop::machines
        JobShop& shop;

        /* Function */
        Machine(string _descr, JobShop& _shop, Sim& _sim);
        SimCoroutine run() override;
    };

    /*********************************************************/
    /* ----------------- DERIVED CLASS --------------------- */
    /*********************************************************/

    // DECISION MAKER: Handles the "activity" to "machine" dispatching process
    class Dispatcher : public SimObject {
    public:
        /* Attributes */
        JobShop& shop;
        bool random_choice;
        SimCondWaiting* condition;
        vector<Machine*> free;
        SimTime pt;
        
        /* Function */
        Dispatcher(std::string _descr, Sim& _sim, JobShop &_shop);
        ~Dispatcher() {};
        SimCoroutine run() override;
        bool is_ready();
        Machine* choose_machine(Activity* activity);
        bool process(Activity& activity);
        void process_next_ready_activity();
        void process_ready_activity();
        void process_activity(Activity* activity);
        void get_possible_actions(vector<Action> &actions);
    };

    /*********************************************************/
    /* ----------------- DERIVED CLASS --------------------- */
    /*********************************************************/

    // Handles scheduled events, i.e., job & activity arrivals and completions
    class OperationsManager : public SimObject {
    public:
        /* Attributes */
        JobShop& shop;

        /* Function */
        OperationsManager(std::string _descr, Sim& _sim, JobShop& _shop);
        ~OperationsManager() {};
        SimCoroutine run() override;
    };


    /*********************************************************/
    /* ----------------- DERIVED CLASS --------------------- */
    /*********************************************************/

    // The (start, end) time for each activity and the index of the assigned machine
    typedef std::unordered_map<std::string, tuple<SimTime, SimTime, int> > Schedule; 

    // A vector for each machine
    typedef std::unordered_map<std::string, vector<Activity*> > MachineSequences;
  
    // An activity to machine assignment record
    typedef pair<Activity*, Machine*> Assignment;

    class JobShop : public System {
    public:
        using PriorityOrder = vector<Assignment>; 
       
        /* Attributes */
        vector<Job*> jobs;
        vector<Machine*> machines;
        vector<Activity*> activities;
        OperationsManager* manager;
        Dispatcher* dispatcher;
        vector<Assignment> feasible_assignments; 
        std::unordered_map<string, int> assignment_index;
        SimTime cmax;
        queue<Activity*> ready_to_process;        

        /* Function */

        /// Actions & Reinf. Learning 
        void apply_action(SystemState& state, Action& action) override;
        void get_available_actions(SystemState& state, Actions &actions) override;
        int hash_assignment(Activity* _activity, Machine* _machine);
        string hash_action(Action& action) override;
        Action unhash_action(const string& code) override;
        int get_action_index(Action& action) override;
        void define_action_space();        
        Action get_action(const string& descr) override;
        Action get_action(int a) override;
        Assignment get_assignment(Action& action);
        int find_action(const string& info) override;
        optional<SimTime> get_next_decision_point(SystemState& state, bool exclude_current=false) override;
        void transition_to_time(SystemState& state, SimTime t) override;
        int number_of_actions() override;
        void report_actions(ostream& os);
        void report_action(ostream &os, Action &action) override;
        void record_assignment(SimTime t, Activity* activity, Machine* machine);
        float compute_reward(SystemState& previous_state, SystemState& next_state, Action action) override;
        float reward_function_incr(SystemState& previous_state, SystemState& next_state, Action action);
        float reward_function_incum(SystemState& previous_state, SystemState& next_state, Action action);
        float reward_function_delay(SystemState& previous_state, SystemState& next_state, Action action);
        float compute_projected_cmax(SystemState& state);       

        /// General
        JobShop(Sim &_sim);
        ~JobShop();
        void clear_queue();
        void dequeue_activity(Activity* activity);
        void enqueue_activity(Activity* activity);
        void copy(const System &system) override;
        std::unique_ptr<System> clone(Sim &_sim) const override;
        void copy(const JobShop& shop);
        void schedule_first_activity(Job *job);
        void simulate(SimResult& result);
        float evaluate_actions(vector<Action>& actions, bool stoch=false) override;
        void evaluate_schedule(Schedule& schedule, Metrics &metrics);
        void evaluate(PriorityOrder& order, Metrics& metrics, bool stoch);
        void evaluate_deter(PriorityOrder& order, Metrics& metrics);
        void evaluate_stoch(PriorityOrder& order, Metrics& metrics, bool par_eval);
        void evaluate_expected_performance(PriorityOrder& order, Metrics& metrics, int nb_scen = 1);
        void evaluate_expected_performance_parallel(PriorityOrder& order, Metrics& metrics, int nb_scen = 1);
        void evaluate_expected_performance_stable(PriorityOrder& order, Metrics& metrics, float tol);
        int find_activity(PriorityOrder& order, Activity* activity);
        void load_order(string filename, PriorityOrder &order);

        /// Logging
        void log_gantt(Schedule& schedule, ostream& os);
        void log_schedule(Schedule& schedule, ostream& os);
        void log_sequences(MachineSequences& sequences, ostream& os);
        void log_sequence(vector<Activity*>& sequence, ostream& os);
        void log_ordering(PriorityOrder &order, ostream& os);
        void log_history(ostream& os) override;
     
        /// Priority Orderings, Schedules
        void extract_priority_order(PriorityOrder &order);
        void get_priority_order(vector<Action>& actions, PriorityOrder &order);
        MachineSequences* extract_sequences(PriorityOrder& order);        
        Schedule* build_schedule(PriorityOrder &_order, bool sample);
        Schedule* build_schedule_reverse(PriorityOrder& _order, SimTime t, bool sample);
        void fix_ordering(PriorityOrder& _order);
        vector<Activity*> get_critical_activities(PriorityOrder& _order);
        void copy_order(PriorityOrder& order, PriorityOrder &copy);

        /// Data 
        void load_instance(const string& name);
        void load_stoch_instance(const string& name);
        void customize_instance(const string& name);
        void save_instance(const string& name);
        void add_machine(Machine* mach);
        void add_job(Job* job);
        void create_machines(int m);
        void create_jobs(int n);
        void define_activity(int job_id, int act_id, vector<pair<int, Parameter*> > &cand);
        void load_schedule();

        /// Access 
        Activity* get_activity(const string& name);
        Machine* get_machine(const string& name);
        Job* get_job(const string& name);
        bool is_job_name(const string& name);
        bool is_activity_name(const string& name);
       
        /// State
        SystemState* create_initial_state() override;
        void update(SystemState& state);
        void get_states();
        list<SimTime>* extract_time_points();
        SystemState extract_state(SimTime t) override;
        SimTime extract_time(const string& code) override;
        void restore_state(SystemState& _state) override;
        string hash_state(SystemState& _state) override;
        SystemState unhash_state(const string& code) override;
        //SystemState* interpolate_state(SimTime t, SystemState* nearest) override;
        void reset() override; 
        bool is_terminal_state(SystemState& state) override;      
        void restore_machine(Machine* machine, const StateRecord& record, SystemState& state);
        void restore_pending_activity(Activity* activity, SystemState& state);
        void restore_in_progress_activity(Activity* activity, SystemState& state);
        void restore_completed_activity(Activity* activity, SystemState& state);
        float get_completed_jobs(SystemState& state);
        float get_completed_operations(SystemState& state);
        void set_activity_states(SystemState& state, Job* job, int current);
    };


    /*********************************************************/
    /* ----------------- INTENSIFIED SA -------------------- */
    /*********************************************************/
  
    class Solution {
    public:
        Solution(JobShop::PriorityOrder& _order, Metrics& _metrics);
        ~Solution(); 
        JobShop::PriorityOrder order; 
        Metrics metrics;
    };

    class ISa {
    public:
        std::unordered_map<string, Solution*> solution_archive; 

        /* Attributes */
        int id;
        std::mt19937& gen;
        JobShop& shop;
        bool terminate;
        bool improved_best;
        float t, rate;
        int runtime;
        float tI = 10.0f; // 10;
        float tF = 0.001f;
        int step, no_impr;
        int sign = 1;
        int temp_steps = 50; // 50;
        int max_iter = 5000;// 10000; 
        int accept;     
        int max_no_impr = 3; // 10;
        Solution *current, *best;
        clock_t start, end;  
        double wall_start, wall_end, wall_time;
        ofstream out;
     
        bool stoch, par_eval;

        /* Function */
        ISa(int id, std::mt19937& gen, JobShop &_shop, Solution* starting, string outfile, bool stoch);
        ~ISa();
        void copy_solution(Solution& solution, Solution &copy);
        void prepare();
        void handle_non_improvement(); 
        bool accept_change(float before, float after);
        void restore_best();
        SimTime evaluate(Solution *solution);        
        void log_solution(Solution *solution, ostream &os, bool sample);        
        void update_best(Solution* solution);
        void evaluate_state_changes();
        bool frozen();
        void revise_temperature_linear() { t -= rate;  step++; }
        void revise_temperature_geometric() { t = t * rate; step++;  }
        void iterate();
        void report_progress();
        void finalise();
        void run();
        
        ///
        void local_search(Solution* solution, bool all);
        void local_search_shift(Solution* solution, int a);
        void local_search_shift(Solution* solution);
        void local_search_exchange(Solution* solution);
        void local_search_machine(Solution* solution);
        void local_search_machine(Solution* solution, int a);
        void perturb_solution(Solution* solution);
        void perturb_shift(Solution* solution, bool crit=false);
        void perturb_shift_advanced(Solution* solution, bool crit=false);
        void perturb_assignment(Solution* solution, bool crit=false);
        void improve(Solution* solution);
        void mutate_solution(Solution* solution);
        void shift(int a, int a_, JobShop::PriorityOrder &order); 
    };

    /*********************************************************/
    /* -------------- PARALLEL INTENSIFIED SA -------------- */
    /*********************************************************/

    class PISA {
    public:
        /* Attributes */
        int nbsa, best_at;
        int runtime;
        vector<ISa*> solvers;
        
        JobShop& shop;
        clock_t start, end;
        double wall_start, wall_end, wall_time;
        bool stoch;
        bool par_eval;

        /* Function */
        PISA(JobShop &_shop, int _nbsa, bool _stoch, bool _par_eval);
        ~PISA();
        void run();
        void clear();
        void share_best();
        int get_best();
        bool has_converged();
        void report_progress();
        void log_solution(string file, bool sample);
    };
}