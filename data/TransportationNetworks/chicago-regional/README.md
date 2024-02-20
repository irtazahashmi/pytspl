# Chicago Regional Network and Demand
A large-scale, detailed representation of the Chicago region. 

## Source
Developed and provided by the Chicago Area Transportation Study (CATS)

## Scenario
FIXME

## Contents

 - `ChicagoRegional_net.tntp` Network
 - `ChicagoRegional_node.tntp` Node coordinates
 - `ChicagoRegional_trips.tntp` Demand

## Dimensions
Zones: 1,790
Nodes: 12,982
Links: 39,018
Trips: 1,360,427

## Units
Time: Minutes
Distance: miles
Cost: cents  [ FIXME in what year? ]

## Generalized Cost Weights
Toll: 0.1 minutes/cent
Distance: 0.25 minutes/mile

## Solutions
Best-known link flows solution with Average Excess Cost of 8.3E-12. Optimal objective function value:  30792611.3864393. 

[Aroon Aungsuyanon](mailto://aroon_jung@yahoo.com) studied the set of equilibrium PAS for this network under various demand levels. The results of his analysis:  
 - [5,617 maps for CS=0.2](https://app.box.com/s/i7zvz5gyw8k5kweps90r)  
 - [11,702 maps for CS=0.1](https://app.box.com/s/erhjbaoljv5094xhd9wr) 
 - [22,500 maps for CS=0.05](https://app.box.com/s/vy53ly09y5wjwn08hqur)

## Known Issues
FIXME translate to Github Issues

Wolfgang Scherr from PTV America:
1. There seem to be discrepancies between "speed" and "free flow travel time" in certain links of the Chicago Regional network. For example, the "Chicago Skyway" (nodes: 8158, 7714, 8184, 8190, 8196, 8246, 8208, ...,2049, 2387) has "speed" of 27MPH while the free flow travel times correspond to 45MPH. As indicated above, the results reported here rely on free flow travel times and not on speeds.

2. There are some bi-directional links which in reality represent mono-directional ramps. The current bi-directional coding creates parallel alternatives to get on or off the freeway/expressway, which leads to a kind of local randomness of the link flow solution at these places. I would recommend coding all ramps as one-ways. Here are a couple of such places, defined by key node numbers:
A) 6652, 2430, 2429
B) 8559, 8563
C) 2196, 2301
D) 7138, 8994
E) 10265, 10253, 11946