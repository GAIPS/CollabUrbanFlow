#!/bin/bash
# python jobs/run.py > logs 2> err
sed -i 's/network = .*/network = grid/g' config/train.config
sed -i 's/agent_type = .*/agent_type = GAT/g' config/train.config
git add config/train.config && git commit -m 'Set config to GAT, GRID, delay, action set'
python jobs/run.py > logs 2> err

sed -i 's/network = .*/network = grid_2_3/g' config/train.config
git add config/train.config && git commit -m 'Set config to GAT, GRID 2x3, delay, action set'
python jobs/run.py > logs 2> err

sed -i 's/network = .*/network = 3_3/g' config/train.config
git add config/train.config && git commit -m 'Set config to GAT, GRID 3X3, delay, action set'
python jobs/run.py > logs 2> err

sed -i 's/network = .*/network = grid_4_3/g' config/train.config
git add config/train.config && git commit -m 'Set config to DQN, GRID 4x3, delay, action set'
python jobs/run.py > logs 2> err

sed -i 's/network = .*/network = 6_6/g' config/train.config
git add config/train.config && git commit -m 'Set config to GAT, GRID 6X6, delay, action set'
python jobs/run.py > logs 2> err

sed -i 's/network = .*/network = grid/g' config/train.config
sed -i 's/agent_type = .*/agent_type = DQN/g' config/train.config
git add config/train.config && git commit -m 'Set config to DQN, GRID, delay, action set'
python jobs/run.py > logs 2> err

sed -i 's/network = .*/network = grid_2_3/g' config/train.config
git add config/train.config && git commit -m 'Set config to DQN, GRID 2x3, delay, action set'
python jobs/run.py > logs 2> err

sed -i 's/network = .*/network = 3_3/g' config/train.config
git add config/train.config && git commit -m 'Set config to DQN, GRID 3X3, delay, action set'
python jobs/run.py > logs 2> err

sed -i 's/network = .*/network = grid_4_3/g' config/train.config
git add config/train.config && git commit -m 'Set config to DQN, GRID 4x3, delay, action set'
python jobs/run.py > logs 2> err

sed -i 's/network = .*/network = 6_6/g' config/train.config
git add config/train.config && git commit -m 'Set config to DQN, GRID 6X6, delay, action set'
python jobs/run.py > logs 2> err
