from cityflow import Engine
import json


config_file_path = 'network/intersection/config.json'
eng = Engine(config_file_path, thread_num=1)

with open('network/intersection/roadnet.json', 'r') as f:
    network = json.load(f)


intersections = [item for item in network['intersections'] if not item['virtual']]
phasectl = [(-1,0)] * len(intersections) # PHASE ID, NEXT CHANGE
for step in range (3600):
    for i, intersection in enumerate(intersections):
        phaseid, next_change = phasectl[i]
        tl = intersection['trafficLight']['lightphases']
        if next_change == step:
            phaseid = (phaseid + 1) % len(tl)
            eng.set_tl_phase(intersection['id'] , phaseid)

            next_change = step + tl[phaseid]['time']
            phasectl[i] = (phaseid, next_change)


    eng.next_step()
    eng.get_current_time()
    eng.get_lane_vehicle_count()
    eng.get_lane_waiting_vehicle_count()
    eng.get_lane_vehicles()
    eng.get_vehicle_speed()
    # do something 
