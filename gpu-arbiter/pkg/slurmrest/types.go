package slurmrest

import "strings"

// NoVal is Slurm's optional number, v0.0.42_uint{16,32,64}_no_val_struct:
// {"set", "infinite", "number"}.
type NoVal struct {
	Set      bool  `json:"set"`
	Infinite bool  `json:"infinite"`
	Number   int64 `json:"number"`
}

// Num is a set number.
func Num(n int64) *NoVal { return &NoVal{Set: true, Number: n} }

// Value returns the number and whether it is set (and finite).
func (n NoVal) Value() (int64, bool) { return n.Number, n.Set && !n.Infinite }

// JobDesc is the part of v0.0.42_job_desc_msg devai submits (C3). Every
// devai job sets requeue false, so the field is always sent.
type JobDesc struct {
	Script                  string   `json:"script"`
	Name                    string   `json:"name,omitempty"`
	Licenses                string   `json:"licenses,omitempty"`
	TimeLimit               *NoVal   `json:"time_limit,omitempty"`
	KillWarningSignal       string   `json:"kill_warning_signal,omitempty"`
	KillWarningDelay        *NoVal   `json:"kill_warning_delay,omitempty"`
	KillWarningFlags        []string `json:"kill_warning_flags,omitempty"`
	Requeue                 bool     `json:"requeue"`
	CurrentWorkingDirectory string   `json:"current_working_directory,omitempty"`
	StandardOutput          string   `json:"standard_output,omitempty"`
	StandardError           string   `json:"standard_error,omitempty"`
	Environment             []string `json:"environment"`
	Comment                 string   `json:"comment,omitempty"`
	Hold                    bool     `json:"hold,omitempty"`
}

// ExitCode is v0.0.42_process_exit_code_verbose.
type ExitCode struct {
	Status     []string `json:"status"`
	ReturnCode NoVal    `json:"return_code"`
	Signal     struct {
		ID   NoVal  `json:"id"`
		Name string `json:"name"`
	} `json:"signal"`
}

// JobInfo is the part of v0.0.42_job_info (slurmctld's view) devai reads.
type JobInfo struct {
	JobID            int64    `json:"job_id"`
	Name             string   `json:"name"`
	UserName         string   `json:"user_name"`
	JobState         []string `json:"job_state"`
	StateReason      string   `json:"state_reason"`
	StateDescription string   `json:"state_description"`
	Comment          string   `json:"comment"`
	AdminComment     string   `json:"admin_comment"`
	Licenses         string   `json:"licenses"`
	Hold             bool     `json:"hold"`
	Requeue          bool     `json:"requeue"`
	Partition        string   `json:"partition"`
	Nodes            string   `json:"nodes"`
	BatchFlag        bool     `json:"batch_flag"`
	RestartCnt       int      `json:"restart_cnt"`
	ExitCode         ExitCode `json:"exit_code"`
	SubmitTime       NoVal    `json:"submit_time"`
	StartTime        NoVal    `json:"start_time"`
	EndTime          NoVal    `json:"end_time"`
	TimeLimit        NoVal    `json:"time_limit"`
	Priority         NoVal    `json:"priority"`
}

// HasState reports whether the job is in state s (job_state is a list).
func (j JobInfo) HasState(s string) bool {
	for _, x := range j.JobState {
		if strings.EqualFold(x, s) {
			return true
		}
	}
	return false
}

// NodeInfo is the part of v0.0.42_node devai reads.
type NodeInfo struct {
	Name            string   `json:"name"`
	State           []string `json:"state"`
	Reason          string   `json:"reason"`
	ReasonSetByUser string   `json:"reason_set_by_user"`
	ReasonChangedAt NoVal    `json:"reason_changed_at"`
	CPUs            int      `json:"cpus"`
	AllocCPUs       int      `json:"alloc_cpus"`
	RealMemory      int64    `json:"real_memory"`
	BootTime        NoVal    `json:"boot_time"`
}

// HasState reports whether the node carries state s (e.g. DRAIN).
func (n NodeInfo) HasState(s string) bool {
	for _, x := range n.State {
		if strings.EqualFold(x, s) {
			return true
		}
	}
	return false
}

// DBJob is the part of v0.0.42_job (slurmdbd's record) devai reads.
type DBJob struct {
	JobID   int64  `json:"job_id"`
	Name    string `json:"name"`
	User    string `json:"user"`
	Comment struct {
		Administrator string `json:"administrator"`
		Job           string `json:"job"`
		System        string `json:"system"`
	} `json:"comment"`
	State struct {
		Current []string `json:"current"`
		Reason  string   `json:"reason"`
	} `json:"state"`
	ExitCode        ExitCode `json:"exit_code"`
	DerivedExitCode ExitCode `json:"derived_exit_code"`
	Licenses        string   `json:"licenses"`
	Hold            bool     `json:"hold"`
	Nodes           string   `json:"nodes"`
	Time            struct {
		Elapsed    int64 `json:"elapsed"`
		Eligible   int64 `json:"eligible"`
		End        int64 `json:"end"`
		Start      int64 `json:"start"`
		Submission int64 `json:"submission"`
		Limit      NoVal `json:"limit"`
	} `json:"time"`
}

// APIError is one entry of a response's `errors`.
type APIError struct {
	Description string `json:"description"`
	ErrorNumber int    `json:"error_number"`
	Error       string `json:"error"`
	Source      string `json:"source"`
}

// response is the envelope every v0.0.42 response shares.
type response struct {
	Errors   []APIError `json:"errors"`
	Warnings []struct {
		Description string `json:"description"`
		Source      string `json:"source"`
	} `json:"warnings"`
}
