| config                           |   accuracy |   avg_steps |   avg_tool_calls |   invalid_reply_rate |   tool_error_rate |   avg_seconds_per_task |   acc_arithmetic |   acc_sql |   acc_units |   acc_policy |   acc_multi_tool |
|:---------------------------------|-----------:|------------:|-----------------:|---------------------:|------------------:|-----------------------:|-----------------:|----------:|------------:|-------------:|-----------------:|
| Qwen2.5-1.5B - direct (no tools) |      0.067 |        1    |             0    |                0     |             0     |                    7.2 |            0.167 |         0 |       0     |          0.2 |                0 |
| Qwen2.5-1.5B - agent + tools     |      0.5   |        3.1  |             1.87 |                0.129 |             0.321 |                   89.6 |            0.333 |         1 |       0.667 |          0.2 |                0 |
| Qwen2.5-0.5B - agent + tools     |      0.033 |        5.57 |             1.53 |                0.701 |             0.174 |                   68.3 |            0     |         0 |       0.167 |          0   |                0 |

| config                           |   answered without tools |   correct |   invalid tool-call format |   ran out of steps |   tool error not recovered |   wrong answer after tool use |
|:---------------------------------|-------------------------:|----------:|---------------------------:|-------------------:|---------------------------:|------------------------------:|
| Qwen2.5-0.5B - agent + tools     |                        0 |         1 |                          2 |                 26 |                          0 |                             1 |
| Qwen2.5-1.5B - agent + tools     |                        0 |        15 |                          1 |                  5 |                          5 |                             4 |
| Qwen2.5-1.5B - direct (no tools) |                       28 |         2 |                          0 |                  0 |                          0 |                             0 |
