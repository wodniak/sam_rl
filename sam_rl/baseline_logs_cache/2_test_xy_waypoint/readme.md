28.02.2022
Simple XY waypoint test with TD3 model (trained on 1M samples)
Run with commit `xy waypoint test`
Env state 12d
Control with `[rpm, dr]`

To run with stonefish
1. start the simulation with `rosrun sam_stonefish_sim bringup.sh` and start the nodes.
2. Check the paths for `2_test_xy_waypoint/td3_999750_steps.zip`
2. Start RL node `python3 src/trainer_stonefish.py --model td3`

To run with eom env:
1. Check the paths
2. `python3 src/trainer_baseline.py --model td3`