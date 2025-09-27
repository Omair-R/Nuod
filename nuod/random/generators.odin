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


RandomState64 :: struct{
	x: u64,
}

RandomState128 :: struct{
	x: u64,
	y: u64,
}

inner_rand_proc :: proc(
	$T: typeid,
	$RS: typeid,
	init_f : proc(rs: ^RS, seed: u64),
	read_f : proc "contextless" (rs: ^RS) -> T,
	data: rawptr,
	mode: rt.Random_Generator_Mode,
	p: []byte
){
	@(thread_local)
	global_rand_seed : RS

	rs : ^RS = ---

	if data == nil {
		rs = &global_rand_seed 
	} else {
		rs = (cast(^RS)data)
	} 

	switch mode {
		case .Read:
			if rs.x == 0 {
				init_f(rs, 0)
			}

			switch len(p){
				case size_of(T):
					intrinsics.unaligned_store(cast(^T)raw_data(p), read_f(rs))
				case:
					handle_arbitrary_read(T, rs, read_f, p)					
			}
		case .Reset:
			seed: u64
			init_f(rs, get_seed(p))

		case .Query_Info:
			handle_query_info(p, {.Uniform, .Resettable})
	}
}


PCGRandomState :: RandomState64

pcg_random_proc :: proc(
	data: rawptr,
	mode: rt.Random_Generator_Mode,
	p: []byte
) {

	init :: proc (rs: ^RandomState64, seed:u64) {
		rs.x = u64(0)
		_ = read_u32(rs)
		rs.x += seed
		_ = read_u32(rs)
	}

	read_u32 :: proc "contextless"(rs : ^RandomState64) -> u32 {
		t := rs.x
		rot := cast(u32)(rs.x >> 59)
		rs.x = t * u64(6364136223846793005) + u64(1442695040888963407)
		t ~= t >> 18
		
		return rotr32(cast(u32)(t>>27), rot)	
	}

	inner_rand_proc(u32, RandomState64, init, read_u32, data, mode, p)
}


pcg_random_generator :: proc "contextless" (rs : ^PCGRandomState = nil) -> rt.Random_Generator {
	return {
		procedure = pcg_random_proc,
		data = rs,
	}
}


SplitMix64RandomState :: RandomState64

splitmix_random_proc :: proc(
	data: rawptr,
	mode: rt.Random_Generator_Mode,
	p: []byte
) {

	init :: proc (rs: ^SplitMix64RandomState, seed:u64) {
		rs.x = seed
	}

	read_u64 :: proc "contextless"(rs : ^SplitMix64RandomState) -> u64 {
		rs.x += 0x9e3779b97f4a7c15
		tx := rs.x
		tx = (tx ~ tx >> 30) * 0xbf58476d1ce4e5b9
		tx = (tx ~ tx >> 27) * 0x94d049bb133111eb
		return tx ~ tx >> 31		
	}
	
	inner_rand_proc(u64, RandomState64, init, read_u64, data, mode, p)
}


splitmix64_random_generator :: proc "contextless" (rs : ^SplitMix64RandomState = nil) -> rt.Random_Generator {
	return {
		procedure = splitmix_random_proc,
		data = rs,
	}
}



Xorshift64StarRandomState :: RandomState64

xorshift64star_random_proc :: proc(
	data: rawptr,
	mode: rt.Random_Generator_Mode,
	p: []byte
) {

	init :: proc (rs: ^Xorshift64StarRandomState , seed:u64) {
		split_state := SplitMix64RandomState{
			x=seed
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

	inner_rand_proc(u64, RandomState64, init, read_u64, data, mode, p)
}


xorshift64star_random_generator :: proc "contextless" (rs : ^Xorshift64StarRandomState = nil) -> rt.Random_Generator {
	return {
		procedure = xorshift64star_random_proc,
		data = rs,
	}
}


Xorshift128PlusRandomState :: RandomState128

xorshift128plus_random_proc :: proc(
	data: rawptr,
	mode: rt.Random_Generator_Mode,
	p: []byte
) {

	init :: proc (rs: ^Xorshift128PlusRandomState , seed:u64) {
		split_state := SplitMix64RandomState{
			x=seed
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

	inner_rand_proc(u64, RandomState128, init, read_u64, data, mode, p)
}


xorshift128plus_random_generator :: proc "contextless" (rs : ^Xorshift128PlusRandomState = nil) -> rt.Random_Generator {
	return {
		procedure = xorshift128plus_random_proc,
		data = rs,
	}
}


