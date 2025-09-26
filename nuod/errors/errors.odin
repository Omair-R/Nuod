package nuod

import "core:log"
import "core:fmt"
import "core:os"
import "core:strings"

NuodWarning :: enum {
	None,
	UninitializedArrayWarning,
	OperationSkipped,
}

NuodError :: enum {
	None,
	AllocationError,
	ArguementError,
	ArithmeticError,
	NotImplemented,
}


get_default_error_msg :: proc(err: NuodError) -> string {
	switch err {
		case .None:
			return "Not an error."
		case .AllocationError:
			return "An error during allocation."
		case .ArguementError:
			return "Incorrect Arguement."
		case .ArithmeticError:
			return "An error during arithmetic computation."			
		case .NotImplemented:
			return "This method or portion of it, is not implemented yet."
	}
	return ""
}


get_default_warning_msg :: proc(warn: NuodWarning) -> string {
	switch warn {
		case .None:
			return "Not an error."
		case .UninitializedArrayWarning:
			return "Received a none array, nothing has happened."
		case .OperationSkipped:
			return "The array provided has been returned unchanged."
	}
	return ""
}
