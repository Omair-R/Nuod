package random

import "core:fmt"

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


handle_query_info :: proc(p: []byte, info_config: rt.Random_Generator_Query_Info ){
	if len(p) != size_of(rt.Random_Generator_Query_Info) {
		return
	}
	info := (^rt.Random_Generator_Query_Info)(raw_data(p))
	info^ += info_config
}


quick_read_ptr :: proc(rg: rt.Random_Generator, p: rawptr, len: uint) {
	rg.procedure(rg.data, .Read, ([^]byte)(p)[:len])
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
	pcg_rand_seed: PCGRandomState

	rs : ^PCGRandomState = ---

	if data == nil {
		rs = &pcg_rand_seed	
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
			handle_query_info(p, {.Uniform, .Resettable})
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
			handle_query_info(p, {.Uniform, .Resettable})
	}
}


splitmix64_random_generator :: proc "contextless" (rs : ^SplitMix64RandomState = nil) -> rt.Random_Generator {
	return {
		procedure = splitmix_random_proc,
		data = rs,
	}
}



Xorshift64StarRandomState :: struct {
	x : u64,
}


xorshift64star_random_proc :: proc(
	data: rawptr,
	mode: rt.Random_Generator_Mode,
	p: []byte
) {

	init :: proc (rs: ^Xorshift64StarRandomState , seed:u64) {
		split_state := SplitMix64RandomState{
			state=seed
		}
		split_gen := splitmix64_random_generator(&split_state)

		x: u64
		quick_read_ptr(split_gen, &x, size_of(x))
		rs.x = x
	}

	read_u64 :: proc "contextless"(rs : ^Xorshift64StarRandomState) -> u64 {
		rs.x ~= rs.x >> 12
        rs.x ~= rs.x >> 25
        rs.x ~= rs.x >> 26
        return rs.x * u64(0x2545F4914F6CDD1D)
	}

	@(thread_local)
	xorshift64star_rand_seed: Xorshift64StarRandomState

	rs : ^Xorshift64StarRandomState = ---

	if data == nil {
		rs = &xorshift64star_rand_seed	
	} else {
		rs = (cast(^Xorshift64StarRandomState)data)
	} 

	switch mode {
		case .Read:
			if rs.x == 0 {
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
			handle_query_info(p, {.Uniform, .Resettable})
	}
}


xorshift64star_random_generator :: proc "contextless" (rs : ^Xorshift64StarRandomState = nil) -> rt.Random_Generator {
	return {
		procedure = xorshift64star_random_proc,
		data = rs,
	}
}


Xorshift128PlusRandomState :: struct {
	x : u64,
	y : u64,
}


xorshift128plus_random_proc :: proc(
	data: rawptr,
	mode: rt.Random_Generator_Mode,
	p: []byte
) {

	init :: proc (rs: ^Xorshift128PlusRandomState , seed:u64) {
		split_state := SplitMix64RandomState{
			state=seed
		}
		split_gen := splitmix64_random_generator(&split_state)

		x: u64
		quick_read_ptr(split_gen, &x, size_of(x))
		rs.x = x
		quick_read_ptr(split_gen, &x, size_of(x))
		rs.y = x
	}

	read_u64 :: proc "contextless"(rs : ^Xorshift128PlusRandomState) -> u64 {
		tx := rs.x
		ty := rs.y

		rs.x = ty

		tx ~= tx << 23
		tx ~= tx >> 18
		tx ~= ty ~ ty >> 5

		rs.y = tx

		return tx + ty
	}

	@(thread_local)
	xorshift128plus_rand_seed: Xorshift128PlusRandomState

	rs : ^Xorshift128PlusRandomState = ---

	if data == nil {
		rs = &xorshift128plus_rand_seed	
	} else {
		rs = (cast(^Xorshift128PlusRandomState)data)
	} 

	switch mode {
		case .Read:
			if rs.x == 0 || rs.y == 0 {
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
			handle_query_info(p, {.Uniform, .Resettable})
	}
}


xorshift128plus_random_generator :: proc "contextless" (rs : ^Xorshift128PlusRandomState = nil) -> rt.Random_Generator {
	return {
		procedure = xorshift128plus_random_proc,
		data = rs,
	}
}


