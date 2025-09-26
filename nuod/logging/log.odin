package logging

import "core:log"
import "core:fmt"
import errors "../errors"


error :: proc(err: errors.NuodError, msg:="", location:=#caller_location){
	msg:= msg

	if msg == ""{
		msg = errors.get_default_error_msg(err)
	}
	
	msg = fmt.tprintf("%s:  %s", err, msg)
	log.error(msg, location=location)
}

warning :: proc(warn: errors.NuodWarning, msg:="", location:=#caller_location){
	msg:= msg

	if msg == ""{
		msg = errors.get_default_warning_msg(warn)
	}
	
	msg = fmt.tprintf("%s:  %s", warn, msg)
	log.warn(msg, location=location)
}


debug :: log.debug
info :: log.info
fatal :: log.fatal

