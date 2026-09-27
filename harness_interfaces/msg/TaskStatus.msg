# Ground-truth progress of the cell (published by the simulator).
std_msgs/Header header
bool[] forks_routed          # wire held below the lips of fork i
bool connector_seated        # connector sits in its holder
bool success                 # all forks routed and connector seated
float64 max_contact_force    # largest gripper contact force this episode [N]
int64 episode_seed
