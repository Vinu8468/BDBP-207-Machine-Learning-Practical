# Bayesian Network Inference and Bayesian Learning

# P(Cloudy)
# | 0.5 | 0.5 |

# P(Sprinkler | Cloudy)
#   sprinkler
#   F      T
#F|0.5 | 0.5|
#T|0.9 | 0.1|

# P(Rain | Cloudy)
#   rain
#   F     T
#F|0.8 | 0.2|
#T|0.2 | 0.8|

# P(Wet grass | Sprinkler, Rain)
#                   wet grass
# Sprinkler Rain     F     T
#   F        F     | 1  |  0 |
#   F        T     | 0.1| 0.9|
#   T        F     | 0.1| 0.9|
#   T        T     |0.01|0.99|

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# P(Cloudy)
p_cloudy = TabularCPD(variable = "Cloudy",variable_card = 2,values = [[0.5],[0.5]])

# P(Sprinkler | Cloudy)
cp_sprinkler = TabularCPD(variable = "Sprinkler",variable_card = 2,values = [[0.5,0.9],[0.5,0.1]],evidence = ["Cloudy"],evidence_card=[2])

# P(Rain | Cloudy)
cp_rain = TabularCPD(variable="Rain",variable_card=2,values=[[0.8,0.2],[0.2,0.8]],evidence=["Cloudy"],evidence_card=[2])

# P(Wetgrass | Sprinkler, Rain)
cp_wetgrass = TabularCPD(variable = "WetGrass",variable_card=2,values = [[1,0.1,0.1,0.01],[0,0.9,0.9,0.99]],evidence=["Sprinkler","Rain"],evidence_card=[2,2])

# print(cp_wetgrass)

# +-------------+--------------+--------------+--------------+--------------+
# | Sprinkler   | Sprinkler(0) | Sprinkler(0) | Sprinkler(1) | Sprinkler(1) |
# +-------------+--------------+--------------+--------------+--------------+
# | Rain        | Rain(0)      | Rain(1)      | Rain(0)      | Rain(1)      |
# +-------------+--------------+--------------+--------------+--------------+
# | WetGrass(0) | 1.0          | 0.1          | 0.1          | 0.01         |
# +-------------+--------------+--------------+--------------+--------------+
# | WetGrass(1) | 0.0          | 0.9          | 0.9          | 0.99         |
# +-------------+--------------+--------------+--------------+--------------+

###### this is soo cool
