import json
import os
import torch.distributed as dist

def read_json(rpath: str):
    result = []
    with open(rpath, 'rt') as f:
        for line in f:
            result.append(json.loads(line.strip()))

    return result
def collect_result(result, rank, epoch):
    with open(f"./temp_result/temp_result_epoch_{epoch}_rank{rank}.json", 'wt') as f:
        for res in result:
            f.write(json.dumps(res) + '\n')
    dist.barrier()

    result = []
    if rank == 0:
        for rank_id in range(dist.get_world_size()):
            result += read_json(f"./temp_result/temp_result_epoch_{epoch}_rank{rank_id}.json")

        result_new = []
        id_list = set()
        for res in result:
            if res["question_id"] not in id_list:
                id_list.add(res["question_id"])
                result_new.append(res)
        result = result_new

        json.dump(result, open(f"./temp_result/temp_result_epoch_{epoch}.json", 'w'), indent=4)
        print(f"../temp_result/temp_result_epoch_{epoch}.json")

    dist.barrier()
    print(len(result))

    return result