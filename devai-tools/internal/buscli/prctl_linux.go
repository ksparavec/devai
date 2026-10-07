package buscli

import "syscall"

// SetParentDeathSignal makes the kernel send sig to this process when its
// parent dies (PR_SET_PDEATHSIG). A hold whose job script was SIGKILLed
// would otherwise be re-parented to tini -s, leave the step's process tree,
// and renew its lease for ever (C3).
func SetParentDeathSignal(sig syscall.Signal) error {
	_, _, errno := syscall.RawSyscall(syscall.SYS_PRCTL, syscall.PR_SET_PDEATHSIG, uintptr(sig), 0)
	if errno != 0 {
		return errno
	}
	return nil
}
