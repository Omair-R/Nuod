package random


import "core:math/rand"
import "base:intrinsics"
import rt "base:runtime"

import md "../mdarray"
import "../logging"


handle_arbitrary_read :: proc(
	$T: typeid,
	random_state: ^$S,
	read_f: proc "contextless" (^S) -> T,
	p: []byte,
){
	pos := i8(0)
	val := T(0)
	for &v in p {
		if pos == 0 {
			val = read_f(random_state)
			pos = size_of(T)
		}
		v = byte(val)
		val >>= 8
		pos -= 1
	}
}


get_seed :: proc "contextless"(
	p: []byte,
) -> u64 {
	seed : u64
	rt.mem_copy_non_overlapping(&seed, raw_data(p), min(size_of(seed), len(p)))
	if seed == 0 {
		seed = u64(intrinsics.read_cycle_counter())
	}
	return seed
}

rotr32 :: proc "contextless"(x: u32, r: u32) -> u32{
	return x >> r | x << (-r & 31)
}


rotr64 :: proc "contextless"(x: u64, r: u64) -> u64{
	return x >> r | x << (-r & 63)
}


rotl32 :: proc "contextless"(x: u32, r: u32) -> u32{
	return x << r | x >> (32 - r)
}


rotl64 :: proc "contextless"(x: u64, r: u64) -> u64{
	return x << r | x >> (64 - r)
}


PCGRandomState :: struct {
	state : u64,
}


pcg_random_proc :: proc(
	data: rawptr,
	mode: rt.Random_Generator_Mode,
	p: []byte
) {

	init :: proc "contextless" (rs: ^PCGRandomState, seed:u64) {
		rs.state = u64(0)
		_ = read_u32(rs)
		rs.state += seed
		_ = read_u32(rs)
	}

	read_u32 :: proc "contextless"(rs : ^PCGRandomState) -> u32 {
		t := rs.state
		rot := cast(u32)(rs.state >> 59)
		rs.state = t * u64(6364136223846793005) + u64(1442695040888963407)
		t ~= t >> 18
		
		return rotr32(cast(u32)(t>>27), rot)	
	}

	@(thread_local)
	global_rand_seed: PCGRandomState

	rs : ^PCGRandomState = ---

	if data == nil {
		rs = &global_rand_seed	
	} else {
		rs = (cast(^PCGRandomState)data)
	} 

	switch mode {
		case .Read:
			if rs.state == 0 {
				init(rs, 0)
			}

			switch len(p){
				case size_of(u32):
					intrinsics.unaligned_store(cast(^u32)raw_data(p), read_u32(rs))
				case:
					handle_arbitrary_read(u32, rs, read_u32, p)					
			}
		case .Reset:
			seed: u64
			init(rs, get_seed(p))

		case .Query_Info:
			if len(p) != size_of(rt.Random_Generator_Query_Info) {
				return
			}
			info := (^rt.Random_Generator_Query_Info)(raw_data(p))
			info^ += {.Uniform, .Resettable}
	}
}


pcg_random_generator :: proc "contextless" (rs : ^PCGRandomState = nil) -> rt.Random_Generator {
	return {
		procedure = pcg_random_proc,
		data = rs,
	}
}


SplitMix64RandomState :: struct {
	state : u64,
}


splitmix_random_proc :: proc(
	data: rawptr,
	mode: rt.Random_Generator_Mode,
	p: []byte
) {

	init :: proc "contextless" (rs: ^SplitMix64RandomState, seed:u64) {
		rs.state = seed
	}

	read_u64 :: proc "contextless"(rs : ^SplitMix64RandomState) -> u64 {
		rs.state += 0x9e3779b97f4a7c15
		tx := rs.state
		tx = (tx ~ tx >> 30) * 0xbf58476d1ce4e5b9
		tx = (tx ~ tx >> 27) * 0x94d049bb133111eb
		return tx ~ tx >> 31		
	}

	@(thread_local)
	split_mix_rand_seed: SplitMix64RandomState

	rs : ^SplitMix64RandomState = ---

	if data == nil {
		rs = &split_mix_rand_seed	
	} else {
		rs = (cast(^SplitMix64RandomState)data)
	} 

	switch mode {
		case .Read:
			if rs.state == 0 {
				init(rs, 0)
			}

			switch len(p){
				case size_of(u64):
					intrinsics.unaligned_store(cast(^u64)raw_data(p), read_u64(rs))
				case:
					handle_arbitrary_read(u64, rs, read_u64, p)					
			}
		case .Reset:
			seed: u64
			init(rs, get_seed(p))

		case .Query_Info:
			if len(p) != size_of(rt.Random_Generator_Query_Info) {
				return
			}
			info := (^rt.Random_Generator_Query_Info)(raw_data(p))
			info^ += {.Uniform, .Resettable}
	}
}


splitmix64_random_generator :: proc "contextless" (rs : ^SplitMix64RandomState = nil) -> rt.Random_Generator {
	return {
		procedure = pcg_random_proc,
		data = rs,
	}
}
