package main

import "core:fmt"
import "core:os"
import "core:strings"
import "core:strconv"
import "core:mem"

/* id,name,base_experience,height,weight,types,abilities,moves,stats */
Pokemon_Type :: enum {
	FIRE,
	GRASS,
	WATER,
	FLYING,
	GROUND,
	ROCK,
	POISON,
	STEEL,
	ELECTRIC,
	FIGHTING,
	FAIRY,
	NORMAL,
	BUG,
	PSYCHIC,
	DARK,
	GHOST,
}

Pokemon_Stats :: struct {
	hp: int,
	attack: int,
	defense: int,
	special_attack: int,
	special_defese: int,
	speed: int,
}
Pokemon :: struct {
	id: int,
	name: string,
	base_experience: int,
	height: int,
	weight: int,
	types: Pokemon_Type,
	abilities: string,
	moves: string ,
	stats: Pokemon_Stats
}

main :: proc() {

	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)
		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not free: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}
	data, ok := os.read_entire_file_from_filename("pokemon_data.csv")
	defer delete(data)

	s := string(data)
	line_count := 0
	pokemon_data := make_dynamic_array([dynamic]Pokemon)
	defer delete(pokemon_data)
	for line in strings.split_lines_iterator(&s) {
		if line_count == 0 {
			line_count += 1
			continue
		}
		line_data := strings.split(line, ",")
		defer delete(line_data)
		curr_data:Pokemon
		curr_data.id = strconv.atoi(line_data[0])
		curr_data.name= line_data[1]
		curr_data.base_experience = strconv.atoi(line_data[2])
		curr_data.height = strconv.atoi(line_data[3])
		curr_data.weight = strconv.atoi(line_data[4])
		curr_data.types = convert_to_type(line_data[5])
		curr_data.abilities = line_data[6]
		curr_data.moves = line_data[7]
		curr_data.stats = convert_to_stats(line_data[8])
		if curr_data.types == .FIRE {
			fmt.println(curr_data)
		}
		line_count += 1

		if line_count == 2 {
			break
		}
	}
}

convert_to_type :: proc(s: string) -> Pokemon_Type {
	data := strings.split(s, ",")
	defer delete(data)
	p_type: Pokemon_Type
	switch data[0] {
	case "fire": p_type = .FIRE
	}
	return p_type
}

convert_to_stats :: proc(s: string) -> Pokemon_Stats {
	stats: Pokemon_Stats
	return stats
}
