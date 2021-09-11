#!/bin/bash
sed -i 's/alpha=0.9/alpha=0.25/g' agents/actor_critic.py
sed -i 's/beta=0.25/beta=0.50/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.25 && beta=0.50 #3'
python jobs/run.py

sed -i 's/beta=0.50/beta=0.75/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.25 && beta=0.75 #3'
python jobs/run.py

sed -i 's/beta=0.75/beta=0.90/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.25 && beta=0.90 #3'
python jobs/run.py

sed -i 's/beta=0.90/beta=0.50/g' agents/actor_critic.py
sed -i 's/alpha=0.25/alpha=0.50/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.50 && beta=0.50 #3'
python jobs/run.py

sed -i 's/alpha=0.50/alpha=0.75/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.75 && beta=0.50 #3'
python jobs/run.py

sed -i 's/alpha=0.75/alpha=0.90/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.90 && beta=0.50 #3'
python jobs/run.py

sed -i 's/alpha=0.90/alpha=0.50/g' agents/actor_critic.py
sed -i 's/beta=0.50/beta=0.75/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.50 && beta=0.75 #3'
python jobs/run.py


sed -i 's/beta=0.75/beta=0.90/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.50 && beta=0.90 #3'
python jobs/run.py

sed -i 's/beta=0.90/beta=0.75/g' agents/actor_critic.py
sed -i 's/alpha=0.50/alpha=0.75/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.75 && beta=0.75 #3'
python jobs/run.py

sed -i 's/alpha=0.75/alpha=0.90/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.90 && beta=0.75 #3'
python jobs/run.py

sed -i 's/beta=0.75/beta=0.90/g' agents/actor_critic.py
git commit -a -m 'Set alpha=0.90 && beta=0.90 #3'
python jobs/run.py
