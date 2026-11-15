#include "simulator.h"

/* String handling */
vector<string> split_string(const string& str, char delimiter) {    
    vector<string> sub_str;
    std::stringstream stream(str);
    string _str;
    while (getline(stream, _str, delimiter)) 
        sub_str.push_back(_str);
    return sub_str;
}

/* Rounding */
string rounded_string(float num) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << num;
    return out.str();
}

/*********************************************************/
/* --------------- COROUTINE CLASS --------------------- */
/*********************************************************/

// See header

/*********************************************************/
/* --------------- COROUTINE PROMISE CLASS ------------- */
/*********************************************************/

void SimPromise::set_object(SimObject& _obj) { ptr_obj = &_obj; }  

suspend_always SimPromise::initial_suspend() { return {}; }

suspend_always SimPromise::final_suspend() noexcept { return {}; }

void SimPromise::unhandled_exception() {}

/* Creates new coroutine object. Initialize a handle to the coroutine's state */
SimCoroutine SimPromise::get_return_object() {
    coroutine_handle<SimPromise> handle = coroutine_handle<SimPromise>::from_promise(*this);
    return SimCoroutine{ handle };
}

/*********************************************************/
/*------------------- SIM EVENT CLASS ------------------ */
/*********************************************************/

SimEvent::SimEvent(Sim *_sim, EventType _type, SimObject* _obj)
    :sim(_sim), type(_type), obj(_obj) {    
    set_state(EventState::PENDING);
}

SimEvent::SimEvent() {
    sim = nullptr; 
    obj = nullptr; 
    handler = nullptr; 
    type = "undefined"; 
    is_dp = false;
    set_state(EventState::PENDING);    
}

SimEvent::~SimEvent() {}
void SimEvent::set_state(EventState _state) { state = _state; }
void SimEvent::set_type(EventType _type) { type = _type; }

void SimEvent::log(ostream& os) const {
    if (obj != nullptr) {
        string text = std::format("\nEVENT<{}::{}>", type, obj->descr);        
    }
}

bool SimEvent::is_pending() const { return (state == EventState::PENDING); }
bool SimEvent::is_triggered() const { return (state == EventState::TRIGGERED); }
bool SimEvent::is_processed() const { return (state == EventState::PROCESSED); }
bool SimEvent::is_aborted() const { return (state == EventState::ABORTED); }

/* Add the resume method of a process as an handler of the event
   If the event is already triggered or aborted, nothing is done. */
bool SimEvent::add_handler(SimObject *_handler) {
    if (is_triggered()) return false; // Do nothing
    if (is_pending()) handler = _handler;
    return true;
}

/* Process the event */
void SimEvent::process() {
    if (is_aborted() or is_processed()) return; // Do nothing

    if(handler != nullptr) handler->resume(); // Awaken the appropriate handler

    set_state(EventState::PROCESSED); // Now, revise the event state
}

/* Abort the event */
bool SimEvent::abort() {
    if (!is_pending()) return false; // Do nothing
    
    set_state(EventState::ABORTED); // Change the event state

    return true;
}

void SimEvent::define(Sim* _sim, EventType _type, SimObject* _obj,  SimObject* _handler) {
    obj = _obj;
    type = _type;
    sim = _sim;
    add_handler(_handler);
}

/*********************************************************/
/*--------------- SIM QUEUED EVENT CLASS --------------- */
/*********************************************************/

SimQueuedEvent::SimQueuedEvent(Sim* _sim,  EventType _type, SimObject* _obj, SimTime _time, int _priority)
    : SimEvent(_sim, _type, _obj), time(_time), priority(_priority) {}

SimQueuedEvent::SimQueuedEvent():SimEvent() {}

void SimQueuedEvent::define(SimTime _time, Sim* _sim, EventType _type, SimObject* _obj, SimObject* _handler, int priority) {
    time = _time;
    this->priority = priority;
    SimEvent::define(_sim, _type, _obj, _handler);
}

/* Trigger the scheduled event */
bool SimQueuedEvent::trigger()
{
    if (!is_pending()) return false;

    if (time == sim->now()) { // Scheduled event is triggered now!
        set_state(EventState::TRIGGERED); // Change state from "PENDING"        
    }

    sim->queue_event(this); // Queue this event

    return true;
}

/* Logging */ 
void SimQueuedEvent::log(ostream& os) const {
    if (obj != nullptr) //  Guard statements
    {
        std::string text = "\n[t=" + std::to_string(time) + "]. EVENT<" + type + "::" + obj->descr + ">";        
        os << text;
    }
}

/*********************************************************/
/*------------ SIM CONDITIONAL EVENT CLASS ------------- */
/*********************************************************/

SimCondEvent::SimCondEvent(Sim* _sim, EventType _type, SimObject* _obj, SimObject* _handler)
    :SimEvent(_sim, _type, _obj), time_met(-1) { add_handler(_handler); }

/*********************************************************/
/*----------------- SIM CONDITION CLASS ---------------- */
/*********************************************************/

SimCondWaiting& await_condition(SimCondWaiting& condition) { return condition; }

std::suspend_always suspend_execution() { return std::suspend_always{}; }

SimCondWaiting::SimCondWaiting(Sim& _sim, ConditionFunc _func, bool _delete_after_trigger)
    : sim(_sim), func(_func), delete_after_trigger(_delete_after_trigger) {
    sim.add_condition(this);
}

bool SimCondWaiting::is_met() { return func(); }

/* Trigger a waiting coroutine */
bool SimCondWaiting::trigger() {
    if (is_met()) { // Triggering condition is met
        if (handle) handle.resume();// Resume the coroutine that is waiting
        return true;
    }  
    return false;
}

/* Checks whether the coroutine should continue immediately or suspend */
bool SimCondWaiting::await_ready() const { return func(); }

/* Activated when co_await is called to pause the coroutine */
void SimCondWaiting::await_suspend(coroutine_handle<SimPromise> _handle) { handle = _handle; }

/* Called when the coroutine resumes, after the awaited operation is completed */
void SimCondWaiting::await_resume() {}

/*********************************************************/
/*----------------------- CLASS ------------------------ */
/*********************************************************/

/* Preallocate event objects */ 
Sim::Sim(int _id) 
    : time_now(0), max_time(infinity), id(_id), event_pool(max_events) {
    next_event = nullptr; // Define no current event
    next_event_id = 0;
    system = nullptr;    
}

/* Assumption: SimObjects are created outside the Sim and are not deleted here! */
Sim::~Sim() {
    
    for (auto event : cond_events) delete event;
    for (auto condition : conditions) delete condition;
    
    delete_event_queue();
}

SimObject* Sim::find_object(const string &name) { return objects[name]; }

SimTime Sim::now() { return time_now; }
void Sim::set_decision_interface(DecisionInterface& interface) { decision_interface.emplace(interface); }
void Sim::set_now(SimTime t) { time_now = t; }
int Sim::get_event_id() { return next_event_id++; }// Return current value and then increment
void Sim::set_max_time(SimTime _max_time) { max_time = _max_time; }
void Sim::queue_event(SimQueuedEvent *ev) { event_queue.push(ev); }
void Sim::dequeue_next_event() { event_queue.pop(); }
void Sim::log_event(ostream& os, SimQueuedEvent& event) { event.log(os); }
bool Sim::event_queue_empty() { return event_queue.empty(); }
bool Sim::not_event_queue_empty() { return !event_queue.empty(); }
void Sim::record_to_event_log(ostream &os, string info) { os << "\n" << info; }

void Sim::delete_event_queue() {
    while (!event_queue.empty()) {
        SimQueuedEvent* event = event_queue.top();
        event->set_state(EventState::PENDING);
        event_queue.pop(); 
        event_pool.release(event);//delete event; 
    }
}

/* Object pooling approach */
void Sim::schedule_event(SimTime t, EventType type, SimObject* obj, SimObject* handler, int priority) {
    SimQueuedEvent* event = event_pool.acquire();
    if (event == nullptr) throw runtime_error("Event pool is empty");
    event->define(t, this, type, obj, handler, priority);
    event->trigger();
}

/* Basic approach */
void Sim::basic_schedule_event(SimTime t, EventType type, SimObject* obj, SimObject* handler, int priority) {
    SimQueuedEvent* event = new SimQueuedEvent(this, type, obj, t, priority);
    event->add_handler(handler);
    event->trigger();    
}

/* Simplification */
void Sim::schedule_event(SimTime t, SimObject* obj, int priority) { 
    schedule_event(t, nullptr, obj, obj, priority); 
}

/* Return the time of the next event if one exists  */
optional<SimTime> Sim::peek_next_time() {
    if (auto* event = get_next_event()) 
        return event->time;
    return std::nullopt;
}

/* This function will only identify queued_events. Conditional 
events are invisible until they are triggered */
SimQueuedEvent* Sim::get_next_event() {
    return event_queue.empty() ? nullptr : event_queue.top();
}

void Sim::display_queued_events(ostream &os) {
    priority_queue<SimQueuedEvent*, vector<SimQueuedEvent*>, CompareSimQueuedEvent > copy_events;
    while (!event_queue.empty()) {
        SimQueuedEvent* next_event = get_next_event();
        if (next_event) {
            os << "\nEvent: " << next_event->obj->descr << "(" << next_event->priority << ")";
            dequeue_next_event();
            copy_events.push(next_event);
        }
    }
    event_queue = copy_events;  // Reinstate events queue       
}

void Sim::advance_to_next_dp() {
    while (true) {
        next_event = get_next_event(); // Extract next event from the event queue
        if (!next_event->is_dp) advance();
    }
}

/* Assumption: Previously checked that there are queued scheduled events */
void Sim::process_next_scheduled_event() {
    steps++;
    //cout << "\nNext scheduled event processed!";
    next_event = get_next_event(); // Identify the next event from the event queue
    time_now = next_event->time; // Set current time as time of the event 
    next_event->process(); // Process the event
    dequeue_next_event(); // Remove event from the queue
    // _____________________________________________________________________
    /// delete next_event; // Code needed if using "basic_schedule_event" 
    /// Otherwise use:
    /// --------------------------------------------------------------------
    next_event->set_state(EventState::PENDING); // Reset state
    event_pool.release(next_event); // Send back to the pool
}

/* */
void Sim::step(Action& action) // NEW FUNCTIONALITY
{
    //system->apply_action(some_state, action);
    // ?

}

/* A simulation step - process the next scheduled event and/or trigger conditonal events */
// State recording ommited as computationally slow & memory intensive
bool Sim::advance() {  
    process_cond_events(); // Update system. This may add or remove events
    if (!has_processable_event()) return false;
    process_next_scheduled_event();    
    return true;
}

// Check work remains and time bound not violated
bool Sim::has_processable_event() {
    return !event_queue_empty() && peek_next_time() <= max_time;
}

/* This code avoids potential issues of early 
   termination at the very start of a simulation run 
bool Sim::can_advance(void) {
    if (not_event_queue_empty()) return true;
    process_cond_events();
    return !event_queue_empty();
}*/


/* Run the simulation  */
/* If no step limit is supplied, !max_steps is true, so the loop continues while advance() succeeds.
   If a limit is supplied, continue while steps < *max_steps and advance() succeeds.
*/
void Sim::run(optional<int> max_steps) {
    steps = 0; // Reset    
    bool status = (!max_steps);
    while ((!max_steps || steps < *max_steps) and advance()) {}    
}

/* Run the simulation to a particular event */
bool Sim::advance_to(SimQueuedEvent &event) {
    while (event.is_pending() and not_event_queue_empty()) advance();
    return event.is_triggered();
 }

void Sim::advance_by(SimTime dur) {
    SimTime target = time_now + dur;
    while (not_event_queue_empty()){
        optional<SimTime> next_time = peek_next_time();
        if (!next_time) break;
        advance(); 
    }
    time_now = target;
}

/* Check whether conditional events occur */
void Sim::process_cond_events() { 
    for (auto event : cond_events) {
        event->trigger();
    }
   
    for (auto condition : conditions) {
        bool triggered = condition->trigger();
    }
   
   // ___________________________________________________
   /* Extension:
   for (auto it = conditions.begin(); it != conditions.end(); it++) {
        SimCondition* condition = *it;
        if (condition->trigger()) {
            if (condition->delete_after_trigger) {
                it = conditions.erase(it);
                delete condition;
            }
        }        
    }*/
}

void Sim::add(SimObject* obj) { objects.emplace(obj->descr, obj);  }
void Sim::add_cond_event(SimCondEvent *event) { cond_events.push_back(event); }
void Sim::add_condition(SimCondWaiting* condition) { conditions.push_back(condition); }

/*********************************************************/
/* ----------------- SIMOBJECT CLASS ------------------- */
/*********************************************************/

SimObject::SimObject(string _descr, Sim& _sim)
    :sim(_sim), descr(_descr) {
    sim.add(this); 
}

/* Copy constructor — deep copy of state */
SimObject::SimObject(const SimObject& other): descr(other.descr), sim(other.sim) {
    //state.set(other.state);    
}

/* Assignment operator */
SimObject& SimObject::operator=(const SimObject& other) {
    if (this != &other) {
        descr = other.descr;
        //state.set(other.state);
    }
    return *this;
}

void SimObject::activate() {  
    coroutine = run();   
    handle = coroutine->get_handle();
    handle.promise().set_object(*this); 

    // Note: This cannot be done in the constructor
    // and must be done before the coroutine is started!
    
    resume(); // Method to activate coroutine
}

void SimObject::resume() {
    if (!handle || handle.done()) {
        throw std::runtime_error("Coroutine handle is not valid or already done.");
    }
    handle.resume();  // Resume coroutine
}

void SimObject::set_state(ObjState _state) { sim_state = _state; }

//PC: added
std::suspend_always SimObject::delay(SimTime duration) {
    sim.schedule_event(sim.time_now + duration, this);
    return suspend_execution();
}

//PC: added
std::suspend_always SimObject::sleep() {
    assert(!asleep);
    asleep = true;
    return suspend_execution();
}

//PC: added
void SimObject::wakeup() {
    if (!asleep) return;
    asleep = false;
    resume();
}

//PC: added
SimCondWaiting& SimObject::wait_until(SimCondWaiting::ConditionFunc func) {
    SimCondWaiting* condition = new SimCondWaiting(sim, func, true);
    return *condition;
}


/*********************************************************/
/* -------------------- STATE CLASS -------------------- */
/*********************************************************/

StateRecord::StateRecord() {
    obj = nullptr; 
    state = "UND";
}

StateRecord::StateRecord(SimObject* _obj, ObjState _state)
:obj(_obj), state(_state) {}

SystemState::~SystemState() { records.clear(); }

/* A standard copy - the parent and copy are from the same system */
void SystemState::copy(SystemState const & _parent) {
    time = _parent.time;
    records.clear();
    is_terminal = _parent.is_terminal;

    // Deep copy from _parent
    for (const auto& [key, record] : _parent.records) {
        ObjState copy_state = record.state;
        StateRecord copy(record.obj, record.state);
        copy.more_info = record.more_info;
        add(copy);
    }
}

// Insert "in_place" the ObjectRecord to the records map
void SystemState::add(const StateRecord &record) {
    if (records.find(record.obj->descr) != records.end())
        throw std::runtime_error("ObjectRecord already exists");
     
    records.emplace(record.obj->descr, record);
}

void SystemState::add_more_info(const string& key, std::string value) { 
    records[key].more_info = value; 
}

bool SystemState::is_more_info(const string& key) {
    auto it = records.find(key);
    return((*it).second.more_info.length() > 0);
}

std::string SystemState::get_more_info(const string& key) {
    StateRecord *record = lookup_record(key);
    return record->more_info;
}

StateRecord* SystemState::lookup_record(const string& name) {
    auto it = records.find(name);
    return &it->second;
}


void SystemState::log(ostream& os) {
    os << "\n____________________________________________";
    os << "\nSystem State @ t = " << time;
    os << "\n--------------------------------------------";

    for (const auto & [key, record] : records) { // entry.first is a string
        os << "\n" << record.obj->descr << "::"<<record.state;
        if (record.more_info.length() > 0) os << "<" << record.more_info << ">";
    }
}

/*********************************************************/
/* ---------------- STATE HISTORY CLASS ---------------- */
/*********************************************************/

SystemHistory::SystemHistory(System* _system):system(_system) {}

SystemHistory::~SystemHistory() { clear(); }

void SystemHistory::clear() {    
    states.clear();    
    actions.clear();
}

void SystemHistory::log_states(ostream &out) { 
    for (SystemState &state : states) state.log(out); 
}

void SystemHistory::record_state(SystemState &_state) { states.push_back(_state); }

SystemState SystemHistory::operator[](int index) { return states[index]; }

void SystemHistory::record_action(Action &action) { actions.push_back(action); }

void SystemHistory::erase_states(SimTime from) {
    if (states.size() > 0) {
        while (states.back().time > from) states.pop_back();
    }
}

SystemState* SystemHistory::find_nearest_state(SimTime t) {
    SystemState* nearest = nullptr;
    
    // Iterate in reverse: 
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        if ((*it).time <= t) return &(*it);
    }

    return nullptr; // Return last valid state or NULL
}

/*********************************************************/
/* ------------------- SYSTEM CLASS -------------------- */
/*********************************************************/

System::System(Sim& _sim):sim(_sim), state_zero(nullptr) {
    sim.system = this;  // Backwards reference
    sim.system->history.system = this; // Backwards reference
}

/* */
void System::record_action(Action& _action) { history.record_action(_action); }

/* */
int System::get_action_index(string hash_action) {
    Action action = get_action(hash_action);
    return get_action_index(action);
}

/* Convert the object records */
void System::set_initial_state(SystemState& initial_state) {       
    state_zero->is_terminal = initial_state.is_terminal;
    state_zero->time = initial_state.time;
    // NOTE: record is a unordered_map<std::string, ObjectRecord> 
    for (auto record : initial_state.records) {          
        state_zero->records[record.first].obj = sim.find_object(record.first);   
        state_zero->records[record.first].more_info = record.second.more_info;       
    }
}

/* Convert action objects to index and time pairs */
vector<pair<int, SimTime>> System::convert_actions(vector<Action>& actions) {
    vector<pair<int, SimTime>> vect;
    for (auto& action : actions) {
        int index = get_action_index(action);
        vect.emplace_back(index, action.when);
    }
    return vect;
}

// Assumption: System reset or valid current system state is required
void System::simulate(SimResult& result, optional<int> steps) { 
    sim.run(steps); 
    result.sim_end = sim.now();
    result.actions_log = convert_actions(history.actions);
}

void System::simulate_from(SystemState& from_state, SimResult& result, std::optional<int> steps) {
    restore_state(from_state);
    simulate(result, steps);
}

/* Asumption: "state" is the current state */
void System::simulate_to_next_dp(SystemState& state) {
    sim.advance_to_next_dp();
    state = extract_state(sim.now());
}

/* */
void System::convert_results(SystemState &state, vector<SimResult>& results, vector<Trajectory>& trajectories) {
    trajectories.clear();
    int nbof = static_cast<int>(results.size());
    trajectories.reserve(nbof);
    for (int i = 0; i < nbof; i++) { 
        trajectories.emplace_back(make_trajectory(state, results[i].actions_log, results[i].sim_end));                
    }
}

/* */
void System::setup_components(int nthrds, int seed) {
    components.resize(nthrds);
    for (int n = 0; n < nthrds; ++n) {
        components[n].sim = std::make_unique<Sim>(n);
        components[n].sys = std::unique_ptr<System>(clone(*components[n].sim));       
        components[n].sys->gen.seed(seed + n); //components[n].sys->gen = std::mt19937(seed + n); 
    }
}

void System::delete_components() { components.clear(); }

/* Copy the system state but convert the StateRecords appropriately */
SystemState System::translate(SystemState& state) {
    SystemState copy;
    copy.time = state.time;
    copy.is_terminal = state.is_terminal;
    for (const auto& [key, record] : state.records) {
        string name = record.obj->descr;
        SimObject* sim_obj = sim.find_object(name);
        StateRecord new_record(sim_obj, record.state);
        new_record.more_info = record.more_info;
        copy.add(new_record);
    }
    return copy;
}
 
/* Assumption: components pre-initialized */
vector<Trajectory> System::batch_simulate( int nsim, int nthrds, int steps,  SystemState & from_state) {
    vector<Trajectory> trajectory;
    trajectory.resize(nsim); // One result object per simulation
   
    #pragma omp parallel for schedule(static) num_threads(nthrds)
    for (int i = 0; i < nsim; ++i) {
        int tid = omp_get_thread_num(); // ID of the thread (from 0 to num_threads - 1)
        SimResult result;
        System& thread_sys = *components[tid].sys; // Activate a particular system
        SystemState state = thread_sys.translate(from_state); // Copy and translate the SimObjects
        thread_sys.simulate_from(state, result, 1000);// steps);
        trajectory[i] = thread_sys.make_trajectory(from_state, result.actions_log, result.sim_end);
    }
    return trajectory;
}

/* Extract current state */
SystemState System::extract_state(SimTime t) {
    SystemState _state; 
    _state.time = t;
    for(auto & object: sim.objects) {
        StateRecord record(object.second, object.second->sim_state);
        _state.add(record);
    }
    return _state;
}

/* Record action index, not the action object */
void extract_action_index(System& sys, SimResult& result) {    
    result.actions_log.reserve(sys.history.actions.size());
    for (auto& action : sys.history.actions) {
        int index = sys.get_action_index(action);
        result.actions_log.emplace_back(index, action.when);
    }
}

/* */
SystemState System::get_next_state(Action &action, SystemState &state) {
    SystemState next_state;
    next_state.copy(state);
    apply_action(next_state, action);
    auto next_point = get_next_decision_point(next_state);
    transition_to_time(next_state, *next_point);
    return next_state;
}

/* */
optional<SimTime> System::go_to_next_decision_point(SystemState& state) {
    bool exclude_current = false; // Exclude current decision point
    Actions actions;
    while (auto next_point = get_next_decision_point(state, exclude_current)) {
        transition_to_time(state, *next_point);// Perform state transition                     
        get_available_actions(state, actions); // Get actions achievable for the current state
        if (actions.not_empty()) return next_point;// Exit and return the next decision point 
        exclude_current = true; // Bypass current time point                          
    }
    return nullopt;
}

/* */
optional<SimTime> System::transition(SystemState& current, Action& action) {
    apply_action(current, action); // Update the system state
    auto next_point = go_to_next_decision_point(current); // Revise state
    current.is_terminal = (next_point) ? false : true;
    return next_point;
}

/* Generate the system states after each action,then assemble the (s,a,s') tuples */
Trajectory System::make_trajectory(SystemState &initial_state, vector<pair<int, SimTime> > &actions_log, SimTime sim_end) {
    restore_state(initial_state);// Re-initialize system
    
    size_t sz = actions_log.size();
    Trajectory traj;
    traj.transitions.reserve(sz); // Reserve specific # of records
    traj.time = sim_end; 

    SystemState state;
    state.copy(initial_state);
    string current_hash = hash_state(state);
    SimTime next_time;
    for (size_t a = 0; a < actions_log.size(); ++a) {
        const auto& record = actions_log[a];
        
        /// Package action (index, time) into an action object
        Action action = get_action(record.first);
        action.when = record.second;

        /// Apply the action and transition the system
        SystemState previous = state;  // Record this in order to calculate the reward
        apply_action(state, action);
        next_time = (a + 1 < actions_log.size()) ? actions_log[a + 1].second : sim_end;
        transition_to_time(state, next_time); 

        /// Create and populate a new Transition object
        auto& transitions = traj.transitions.emplace_back();
        transitions.hash_state = current_hash;
        transitions.hash_action = hash_action(action);
        transitions.hash_next_state = hash_state(state);
        transitions.reward = compute_reward(previous, state, action);
        
        current_hash = transitions.hash_next_state;
    }    
    return traj;
}

/* */
void System::report_trajectory(ostream& os, Trajectory &trajectory) {
    os << "\nActions = [";
    int a = 0;
    for (auto& transitions : trajectory.transitions) {
        Action action = get_action(transitions.hash_action);
        action.when = extract_time(transitions.hash_state);
        report_action(cout, action);
        a++;
    }
    os << "]";
}


/*********************************************************/
/* ------------------ TRAJECTORY CLASS ----------------- */
/*********************************************************/

void Trajectory::copy(const Trajectory& parent) {
    time = parent.time;
    transitions = parent.transitions;
    value = parent.value;
}

/*********************************************************/
/* ---------------- DECISION INTERFACE ----------------- */
/*********************************************************/

DecisionInterface::DecisionInterface(DecisionMaker& _dm):dm(_dm) {}

int DecisionInterface::request_action(SystemState& state, SimTime t) { 
    return dm.choose_action(state);    

}

/*********************************************************/
/* ------------------- MEMORY USAGE -------------------- */
/*********************************************************/

size_t get_mem_used(const std::unordered_map<std::string, std::string>& info) {
    size_t total = sizeof(std::unordered_map<std::string, std::string>); // base container size
    for (const auto& [string_1, string_2] : info) {
        total += 2 * sizeof(std::string);
        total += string_1.capacity() * sizeof(char); // string contents (char array)
        total += string_2.capacity() * sizeof(char); // string contents (char array)
    }
    return total;
}

size_t get_mem_used(const std::unordered_map<std::string, StateRecord>& records) {
    size_t total = sizeof(System); // base container size
    for (const auto& [key, record] : records) {
        //cout << endl << sizeof(key) << " + " << key.capacity() * sizeof(char) << " + " << sizeof(std::pair<SimObject*, ObjState>);
        total += sizeof(key);    // string object overhead
        total += key.capacity() * sizeof(char); // string contents (char array)
        total += sizeof(std::pair<SimObject*, ObjState>);
    }
    return total;
}

size_t get_mem_used(const SystemState& state) {
    size_t total = sizeof(SystemState); // base container size
    total += get_mem_used(state.records); 
    return total; 
}

size_t get_mem_used(const SystemHistory& history) {
    size_t total = sizeof(SystemHistory); // base container size
    for (const auto& state : history.states) {
        total += sizeof(SystemState); 
        total += get_mem_used(state); 
    }
    return total;
}

size_t get_mem_used(const System& system) {
    size_t total = sizeof(System); // base container size
    total += get_mem_used(system.history); 
    return total;
}

double to_gigabytes(size_t bytes) {
    size_t BYTES_PER_GB = 1073741824; // 1024^3
    double temp = static_cast<double>(bytes) / static_cast<double>(BYTES_PER_GB);
    return static_cast<double>(bytes) / static_cast<double>(BYTES_PER_GB);
}

double to_megabytes(size_t bytes) {
    size_t BYTES_PER_MB = 1048576; // 1024^2
    return static_cast<double>(bytes) / static_cast<double>(BYTES_PER_MB);
}

