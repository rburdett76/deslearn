#pragma once

/*********************************************************/
/* -------------------- LIBRARIES  --------------------- */
/*********************************************************/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <queue>
#include <thread>
#include <coroutine>
#include <atomic>
#include <chrono>
#include <functional>
#include <vector>
#include <memory>           // For std::unique_ptr
#include <list>
#include <set>
#include <unordered_set>
#include <random>
#include <optional>
#include <algorithm>
#include <any>              // For std::any
#include <cassert>
#include <thread>
#include <omp.h>
#include <stack>
#include <limits>

/*********************************************************/
/* -------------------- DEFINITIONS -------------------- */
/*********************************************************/

using namespace std;

const float infinity = 10000; // std::numeric_limits<float>::infinity();
const int max_events = 10000;
const float max_float = std::numeric_limits<float>::max();

using SimTime = float;         // The simulation time unit

vector<string> split_string(const string& str, char delimiter); // String handling 

using ObjState = string;
using EventType = string;

enum class EventState { PENDING, TRIGGERED, PROCESSED, ABORTED };

string rounded_string(float num);

/*********************************************************/
/* ---------------- BACKGROUND THEORY ------------------ */
/*********************************************************/

/*
    Processes generate and respond to events, influencing the system's 
    state and scheduling new events.
    
    Each process handles its own events and schedules future events 
    based on interactions with other processes.

    When modelling the following must be determined:
    1) Which SimObject are passive and which are active?
    2) What does each SimObject do?
    3) What events must be explicitly modelled
    4) How to ensure progression of simulation time
*/

/*********************************************************/
/* ------------------ REGARDING STATES ----------------- */
/*********************************************************/

/*
    EVENT STATE::PENDING: An event that has not been triggered or aborted
    EVENT STATE::TRIGGERED: An event that has occurred, but not processed
    EVENT STATE:PROCESSED: An event that has been triggered and processed.
    EVENT STATE::ABORTED: An event that has been aborted.

    States: A triggered event refers to an event that is scheduled or initiated 
    based on certain conditions or actions within the simulation. In other words,
    it is an event that has been set to occur at a specific point in time due 
    to some prior action or condition.
    
    A processed event is an event that has already occurred and has been handled or 
    executed by the simulation system. Once an event is processed, it has been acted 
    upon and its effects have been incorporated into the simulation. 

*/

/*********************************************************/
/* ------------------ REGARDING EVENTS ----------------- */
/*********************************************************/

/*
    The rationale of discrete event simulation is to schedule 
    events to occur at future times. For example, when a task 
    needs to occur, a task commencement event is scheduled. 

    Conditional events, are different. Whilst known, the time 
    they occur is unknown. They are triggered conditionally. 
    At that point, a conditional event may become a scheduled event.
*/

/*********************************************************/
/* ------------- REGARDING DECISON POINTS -------------- */
/*********************************************************/

/* 
    Decisions are made at the time of a scheduled event. 
    However, some events do not lead to a decison-making 
    task, at that time.
*/

/*********************************************************/
/* -------------- FORWARD DECLARATIONS ----------------- */
/*********************************************************/

class Sim;              //  Simulator (a.k.a., Simulation)
class SimEvent;
class SimObject;        // (a.k.a., SimProcess)
class SimPromise;
class SimCoroutine;
class SimCondEvent;
class SimResult;

class SystemState;
class System;
class SystemHistory;

class Action;
class Actions;
class Trajectory;
class Transition;

class DecisionInterface;
class DecisionMaker;

/*********************************************************/
/*-------------------- OBJECT POOL --------------------- */
/*********************************************************/

template <typename T>
class ObjectPool {
    
public:
    std::vector<std::unique_ptr<T>> pool; // owns the objects
    std::stack<T*> free_objects;          // available objects
    
    // Constructor: preallocate objects
    ObjectPool(size_t size) {
        for (size_t i = 0; i < size; ++i) {
            pool.push_back(std::make_unique<T>());
            free_objects.push(pool.back().get());
        }
    }

    // Acquire an object from the pool
    T* acquire() {
        if (free_objects.empty()) {
            throw std::runtime_error("EventPool empty");
            //return nullptr; // optional: throw or expand
        }
        if (free_objects.size() < 100) cout << "\nEventPool is running low";
        T* obj = free_objects.top();
        free_objects.pop();
        return obj;
    }

    // Release an object back to the pool
    void release(T* obj) { free_objects.push(obj); }

    // Optional: clear/reset pool
    void reset() {
        while (!free_objects.empty()) free_objects.pop(); 
        for (auto& p : pool) free_objects.push(p.get()); 
    }
};

/*********************************************************/
/* --------------- COROUTINE CLASS --------------------- */
/*********************************************************/

/* Asynchronous computing construct */

class SimCoroutine {
public:
    using promise_type = SimPromise;
    std::coroutine_handle<promise_type> handle;  // handle of type 'promise_type'

    explicit SimCoroutine(coroutine_handle<promise_type> _handle):handle(_handle) {}
    ~SimCoroutine() { if (handle) { handle.destroy(); } }

    // Move constructor - The move constructor transfers ownership of the coroutine
    // handle from one SimCoroutine object (other) to a new one (this), without copying.
    SimCoroutine(SimCoroutine&& other) noexcept
        : handle(other.handle) {
        other.handle = nullptr;
    }

    // Move assignment - This function defines how to move-assign one SimCoroutine into another.
    // noexcept: Declares that this function won’t throw exceptions (good for performance/safety).
    // Returns a reference to *this so assignment can chain (e.g. a = b = c).
    SimCoroutine& operator=(SimCoroutine&& other) noexcept {
        if (this != &other) {
            if (handle)
                handle.destroy(); // cleanup old coroutine
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }

    void resume() { if (handle) handle.resume(); }
    void activate() { resume(); }
    coroutine_handle<SimCoroutine::promise_type> get_handle() const { return handle; }

    // Copying SimCoroutine is forbidden — both via the copy constructor and the copy assignment operator.
    SimCoroutine(const SimCoroutine&) = delete;
    SimCoroutine& operator=(const SimCoroutine&) = delete;   
};

/*********************************************************/
/* --------------- COROUTINE PROMISE CLASS ------------- */
/*********************************************************/

class SimPromise
{
public:
    /* Attribute */
    SimObject* ptr_obj = nullptr; 

    /* Function */
    SimPromise() = default;
    SimCoroutine get_return_object(); 
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception();
    void set_object(SimObject& _obj); 
    
    // Define either return_void or return_value, but not both.
    void return_void() {}
};


/*********************************************************/
/*----------------------- CLASS ------------------------ */
/*********************************************************/

/* The core Simulation Event object */

class SimEvent {
public:
    /* Attribute */
    SimObject* obj;         // Reference to SimObject related to the event
    EventType type;         // Type of event that will occur - application specific
    EventState state;       // The state of the event
    Sim* sim;               // Reference to the simulation engine
    SimObject* handler;     // Object associated with the event
    bool is_dp;             // Flag describes whether event coincides with a decision point
 
    /* Function */
    SimEvent(Sim* _sim, EventType _type, SimObject *_obj);
    SimEvent();
    ~SimEvent();
    void define(Sim *_sim,  EventType _type, SimObject *_obj, SimObject *_handler);
    virtual void log(ostream& os) const;
    void set_state(EventState _state);
    void set_type(EventType _type);
    void process();
    bool abort(); 
    bool add_handler(SimObject* handler);  
    bool is_pending() const;
    bool is_triggered() const;
    bool is_processed() const;
    bool is_aborted() const;
};

/*********************************************************/
/*----------------------- CLASS ------------------------ */
/*********************************************************/

/* An event queued (a.k.a., scheduled) for a future time */

class SimQueuedEvent : public SimEvent
{
public:
    /* Attribute */
    SimTime time;  // Event is scheduled to occur at this time
    int priority;

    /* Function */
    SimQueuedEvent(Sim* _sim,  EventType _type, SimObject* _obj, SimTime _time, int _priority = 0);
    SimQueuedEvent();
    void define(SimTime _time, Sim* _sim, EventType _type, SimObject* _obj, SimObject* _handler, int _priority=0);
    bool trigger();
    void log(ostream& os) const override;
};

/*********************************************************/
/*----------------------- CLASS ------------------------ */
/*********************************************************/

class SimCondEvent : public SimEvent {
public:
    SimTime time_met;
    SimCondEvent(Sim* _sim, EventType _type=nullptr, SimObject* _obj=nullptr, SimObject* _handler=nullptr);
    virtual void trigger() = 0; 
};

/*********************************************************/
/*----------------------- CLASS ------------------------ */
/*********************************************************/

class SimCondWaiting {
public:
    using ConditionFunc = std::function<bool()>;

    /* Attribute */
    ConditionFunc func;                     // Function specifying condition logic
    coroutine_handle<SimPromise> handle;    // Handle to coroutine
    Sim &sim;                               // Reference to the simulation engine
    bool delete_after_trigger;              // Define a non-persisting conditional event
    bool is_dp;
 
    /* Functions */
    SimCondWaiting(Sim& _sim, ConditionFunc _func, bool _delete_after_trigger = false);
    ~SimCondWaiting() {} // Do nothing 
    bool is_met();
    bool trigger();
    bool await_ready() const;
    void await_suspend(coroutine_handle<SimPromise> _handle);
    void await_resume();
};

SimCondWaiting& await_condition(SimCondWaiting& condition);

std::suspend_always suspend_execution();


/*********************************************************/
/*----------------------- CLASS ------------------------ */
/*********************************************************/

/* class for the comparison of SimQueuedEvent objects  */

class CompareSimQueuedEvent {
public:
    CompareSimQueuedEvent() {}

    bool operator()(const SimQueuedEvent* ev1, const SimQueuedEvent* ev2) {
        if (ev1->time == ev2->time) return (ev1->priority < ev2->priority); // Descending order w.r.t. priority
        else return (ev1->time > ev2->time); //  Ascending order w.r.t. time
    }
};

/*********************************************************/
/*------------------------------------------------------ */
/*********************************************************/

class DecisionMaker {
public:
    DecisionMaker() {}
    virtual int choose_action(SystemState &state) = 0; 
};

class DecisionInterface {
public:
    /* Attribute */
    DecisionMaker& dm;
    
    /* Function */
    DecisionInterface(DecisionMaker& _dm);
    int request_action(SystemState& state, SimTime t);    
};


/*********************************************************/
/*------------------ SIMULATION CLASS ------------------ */
/*********************************************************/

/* The simulation engine */

class Sim {
public:
    using EventQueue = priority_queue<SimQueuedEvent*, vector<SimQueuedEvent*>, CompareSimQueuedEvent >;
   
    /* Attribute */
    System* system; 
    std::unordered_map<std::string, SimObject*> objects;// Fast reference to System objects
   
    EventQueue event_queue;             // Scheduled events
    list<SimCondEvent*> cond_events;
    list<SimCondWaiting*> conditions;
    ObjectPool<SimQueuedEvent> event_pool;    
    SimQueuedEvent* next_event;
    SimTime time_now;                   // Current simulation time 
    SimTime max_time;                   // Simulation time limit
    int next_event_id, id;
    int steps = 0;
    
    optional<DecisionInterface> decision_interface; // Optional 

    /* Function */
    Sim(int _id=0); 
    ~Sim(); 
    SimTime now();
    int get_event_id();
    void set_max_time(SimTime _max_time);    
    void run(optional<int> steps);
    void step(Action& action); 
    bool has_processable_event();
    bool advance();
    void advance_to_next_dp(); 
    void set_now(SimTime t);
    void delete_event_queue();
    void advance_by(SimTime dur);
    bool advance_to(SimQueuedEvent& event);
    void queue_event(SimQueuedEvent *event);
    void process_next_scheduled_event();
    void schedule_event(SimTime t, EventType type, SimObject *ref, SimObject *handler, int priority=0);
    void basic_schedule_event(SimTime t, EventType type, SimObject* obj, SimObject* handler, int priority = 0);
    void schedule_event(SimTime t, SimObject* handler, int priority = 0);   // PC: added to offer a simplified event scheduling
    void process_event(SimEvent& event);
    void log_event(ostream& os, SimQueuedEvent& event);
    void dequeue_next_event();
    void display_queued_events(ostream &os);
    bool event_queue_empty();
    bool not_event_queue_empty();
    void copy_queue(list<SimObject*>& lst);
    void record_to_event_log(ostream &os, string record);
    void process_cond_events();
    void add(SimObject* _obj);
    void add_cond_event(SimCondEvent* event);
    void add_condition(SimCondWaiting* condition);
    SimQueuedEvent* get_next_event();
    optional<SimTime> peek_next_time();
    void set_decision_interface(DecisionInterface& interface);
    //sSystemState get_state();
    SimObject* find_object(const string &name);

};

/*********************************************************/
/* ------------------ SIMOBJECT CLASS ------------------- */
/*********************************************************/

/* SimObject is used to model entities and processes */

class SimObject {
public:
    using Handle = coroutine_handle<SimPromise>;

    /* Attribute */
    std::string descr;  // Object description
    Sim& sim;           // Reference to a simulator
    ObjState sim_state; // Transient state (i.e., simulated/simulation) 
    Handle handle;      // Storing "run" coroutine handle
    bool asleep;        // PC: added to support sleep() - indicates object is suspended by call to sleep
    
    std::optional<SimCoroutine> coroutine; // default: empty
    
    /* Function */
    SimObject(string _descr, Sim& _sim);
    virtual ~SimObject() = default;

    // Copy constructor — deep copy of state
    SimObject(const SimObject& other);
    
    // Optional: move constructor and assignment for better performance
    SimObject(SimObject&&) = default;
    SimObject& operator=(const SimObject& other);

    void set_state(ObjState _state);
    void activate(); // Method to activate a coroutine process
    void resume();
    virtual SimCoroutine run() = 0; // Pure virtual function

    std::suspend_always delay(SimTime duration);                    //PC added this for single line delay of processes
    std::suspend_always sleep();                                    //PC added this to safely suspend and resume without risk of premature resume
    void wakeup();                                                  //PC added (resumes if asleep)
    SimCondWaiting& wait_until(SimCondWaiting::ConditionFunc _func);      //PC added for single line conditional waits
};


/*********************************************************/
/* ---------------- STATE RECORD CLASS ----------------- */
/*********************************************************/

class StateRecord {
public:
    /* Attribute */
    SimObject* obj;
    ObjState state;
    std::string more_info;
    
    /* Function */
    StateRecord();
    StateRecord(SimObject* _obj, ObjState _state);
};

/*********************************************************/
/* ---------------- SYSTEM STATE CLASS ----------------- */
/*********************************************************/

/* 
    State of the system at a specific time 
    Note: Each SystemState has StateRecords with specific SimObject.
    Cloning the system will necessitate the translation of those SimObjects
*/

class SystemState {
public:
    /* Attribute */
    SimTime time; 
    bool is_terminal;
    std::unordered_map<std::string, StateRecord> records;

    /* Function */
    SystemState(): time(0), is_terminal(false) {} 
    ~SystemState();

    void add(const StateRecord &record);
    void log(ostream& os);    
    StateRecord* lookup_record(const string &name);

    void copy(SystemState const& _parent);  
    void translate(SystemState const& _state);
    bool is_more_info(const string& key);
    void add_more_info(const string& key, std::string value);
    std::string get_more_info(const string& key); 
};

/*********************************************************/
/* ------------------- ACTION CLASS -------------------- */
/*********************************************************/

class Action {
public:
    /* Attribute */
    std::any info;  // Can be anything
    float reward = 0.0f;  // Default - No reward
    SimTime when = -1;  // Default - None. 
  
    /* Function */
    Action() = default;
    ~Action() = default;

    void add_reward(float _reward) { reward = _reward; }
    
    template<typename T>
    void define(T&& value) { info = std::forward<T>(value); }

    template<typename T>
    const T& get() const { return std::any_cast<const T&>(info); }

    template<typename T>
    bool is() const { return info.type() == typeid(T); }
};


/*********************************************************/
/* ------------------- ACTIONS CLASS ------------------- */
/*********************************************************/

class Actions {
public:
    /* Attribute */
    vector<Action> info; // No specific ordering condition
    
    /* Function */
    Actions() {}
    ~Actions() {}

    int size() { return (int) info.size(); }
    bool empty() { return (size() == 0); }
    bool not_empty() { return size() > 0; }
    void report(ostream& os);
    void clear() { info.clear(); } // for (auto action : info) delete action;
    void add(Action action) { info.push_back(action); }
    Action& operator [](int a) { return info[a]; }
};


/*********************************************************/
/* --------------- STATE HISTORY CLASS ----------------- */
/*********************************************************/

/* Chronology of System states */

class SystemHistory {
public:
    /* Attribute */
    vector<SystemState> states;
    vector<Action> actions;    // Actions that cause changes in system state, i.e., event log
    System* system;
    
    /* Function */
    SystemHistory(System *_system=nullptr);
    ~SystemHistory();
    void log_states(ostream &os);
    void record_state(SystemState &_state);
    void erase_states(SimTime from);  
    void record_action(Action& action);
    void clear();
  
    SystemState* find_nearest_state(SimTime t);
    SystemState operator[](int index);
    SystemState get_last() { return states.back(); }
};

/*********************************************************/
/* ---------------- TRAJECTORY CLASS ------------------- */
/*********************************************************/

class Transition {
public:
    /* Attributes */
    string hash_state;
    string hash_next_state;
    string hash_action;
    float reward;

    /* Functions */
    Transition() {};
    Transition(string _hash_state, string _hash_next_state, string _hash_action, float _reward) :
        hash_state(_hash_state), hash_next_state(_hash_next_state), hash_action(_hash_action), reward(_reward) {
    }

    // Required for vector<SimResult> to store vector<Transition>
    Transition(const Transition&) = default;
    Transition(Transition&&) noexcept = default;
    Transition& operator=(const Transition&) = default;
    Transition& operator=(Transition&&) noexcept = default;

    ~Transition() = default;
};

class Trajectory {
public: 
    /* Attributes */
    vector<Transition> transitions;
    float time = -1.0f; // The time of the last transition
    float value = - infinity;

    /* Functions */
    Trajectory() {}
    void clear() { transitions.clear(); }
    void report(ostream& os);
    void copy(const Trajectory& parent);
};

/*********************************************************/
/* --------------- SIMULATION RESULTS ------------------ */
/*********************************************************/

class SimResult {
public:
    /* Attributes */
    vector<pair<int,SimTime> > actions_log; // Could define pair<Action, SimTime>
    SimTime sim_end;    
   
    /* Functions */
    SimResult() = default;
    SimResult(SimTime _time) :sim_end(_time) {}
    ~SimResult() = default;
   
    SimResult(const SimResult&) = default;
    SimResult(SimResult&&) noexcept = default;
    SimResult& operator=(const SimResult&) = default;
    SimResult& operator=(SimResult&&) noexcept = default;

    int get_first_action() { return (*actions_log.begin()).first; }
};


/*********************************************************/
/* ------------------- SYSTEM CLASS -------------------- */
/*********************************************************/

/* Simulation Components */
struct alignas(64) SimulationComponents {
    std::unique_ptr<System> sys;
    std::unique_ptr<Sim> sim;
};

enum class RewardType { incr, delayed };

class System {
public:
    /* Attributes */
    Sim &sim; // Reference to a simulator
    SystemHistory history;
    SystemState* state_zero;
    RewardType reward;
    std::mt19937 gen;
  
    /* Function */
    System(Sim& _sim);
    virtual ~System() = default;  // ensures derived destructor is called
    
    vector<SimulationComponents> components;
    void setup_components(int nbof, int seed);
    void delete_components();
    vector<Trajectory> batch_simulate(int nbof, int nthrds, int steps,  SystemState &state);
    void convert_results(SystemState &state, vector<SimResult>& results, vector<Trajectory>& trajectories);
    void simulate(SimResult& result, optional<int> steps=nullopt);
    void simulate_from(SystemState& state, SimResult& result, optional<int> steps=nullopt);
    void simulate_to_next_dp(SystemState& state);
    void record_action(Action &_action);
    vector<pair<int, SimTime>> convert_actions(vector<Action> &actions);
    SystemState translate(SystemState& state);

    virtual void reset() = 0; 
    virtual SystemState* create_initial_state() = 0;
    virtual void restore_state(SystemState& state) = 0;
    virtual SystemState extract_state(SimTime t);
    //virtual SystemState* interpolate_state(SimTime t, SystemState* nearest) = 0;
    SystemState get_next_state(Action& action, SystemState &state);
    virtual string hash_state(SystemState &state) = 0;
    virtual SystemState unhash_state(const string& code) = 0;
    virtual string hash_action(Action& action) = 0;
    virtual Action unhash_action(const string& code) = 0;
    virtual SimTime extract_time(const string& code) = 0;
    virtual Action get_action(const string& descr) = 0;
    virtual Action get_action(int index) = 0;
    virtual int find_action(const string& info) = 0;
    virtual void get_available_actions(SystemState& state, Actions& actions) = 0;
    virtual void report_action(ostream& os, Action &action) = 0;
    virtual void report_actions(ostream& os) = 0;
    virtual void apply_action(SystemState& state, Action& action) = 0;
    virtual optional<SimTime> get_next_decision_point(SystemState& state, bool exclude_current = false) = 0;
    virtual int get_action_index(Action& action) = 0;
    int get_action_index(string hash_action);
    virtual void transition_to_time(SystemState& current_state, SimTime t) = 0;
    optional<SimTime> transition(SystemState& current_state, Action& action);
    virtual int number_of_actions() = 0; 
    virtual bool is_terminal_state(SystemState& state) = 0;
    virtual float compute_reward(SystemState& previous_state, SystemState& next_state, Action action) = 0;
    virtual float evaluate_actions(vector<Action>& actions, bool stoch) = 0;
    virtual void log_history(ostream& os) = 0;
    virtual void copy(const System &source) = 0;
    virtual std::unique_ptr<System> clone(Sim &sim) const = 0;
    void set_initial_state(SystemState& initial_state);
    Trajectory make_trajectory(SystemState& _state, vector<pair<int, SimTime> >& actions, SimTime sim_end);
    void report_trajectory(ostream& os, Trajectory& trajectory);

    optional<SimTime> go_to_next_decision_point(SystemState& state);
};

/*********************************************************/
/* ------------------- MEMORY USAGE -------------------- */
/*********************************************************/

size_t get_mem_used(const System& system);
size_t get_mem_used(const SystemHistory& history);
size_t get_mem_used(const SystemState& state);
size_t get_mem_used(const StateRecord& record);
size_t get_mem_used(const std::unordered_map<std::string, StateRecord>& records);
size_t get_mem_used(const std::unordered_map<std::string, std::string> &info);

double to_gigabytes(size_t bytes);
double to_megabytes(size_t bytes);