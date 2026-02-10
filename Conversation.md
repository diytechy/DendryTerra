Claude’s Plan
DendryTerra Implementation Plan
Overview
Create a Terra addon that implements Dendry noise generation, porting the C++ algorithm from C:\Projects\DendryNoise to Java following Terra's addon conventions.

Project Structure

c:\Projects\DendryTerra\
├── build.gradle.kts
├── settings.gradle.kts
├── gradlew.bat / gradlew
├── gradle/wrapper/
│   ├── gradle-wrapper.jar
│   └── gradle-wrapper.properties
├── src/main/
│   ├── java/com/dfsek/terra/addons/dendry/
│   │   ├── DendryAddon.java          # Entry point (AddonInitializer)
│   │   ├── DendrySampler.java        # Core algorithm (implements Sampler)
│   │   ├── DendryTemplate.java       # Config template (@Value annotations)
│   │   ├── DendryReturnType.java     # Enum: DISTANCE, WEIGHTED, ELEVATION
│   │   └── math/
│   │       ├── Point2D.java          # 2D point with operations
│   │       ├── Point3D.java          # 3D point (x,y = position, z = elevation)
│   │       ├── Vec2D.java            # 2D vector with dot, normalize, rotate
│   │       ├── Segment2D.java        # 2D line segment
│   │       ├── Segment3D.java        # 3D line segment with elevation
│   │       ├── MathUtils.java        # lerp, clamp, distance calculations
│   │       └── CatmullRomSpline.java # Spline subdivision for smoothing
│   └── resources/
│       └── terra.addon.yml           # Addon manifest
└── LICENSE

## Note

My only concern with this implementation is the possibility for the Claud generated function to iterate through points multiple time for each single sample point.  Caching points through the sampler itself or forcing smaller grid windows might be necessary, but it's not clear to me if this is truly necessary without familiarizing with other addons.

After reviewing generated code, there are multiple notes:

Multiple action items:


2. When the Dendry sampler is provided, all x,z points in the tool will be passed through the Dendry sampler, please create a flow diagram that shows how functions from DendrySampler.java are stepped through and what conditions might cause some functions to be skipped or used cached results to ultimately provide a result back to the Terra tool.  Add relevant comments into DendrySampler.java for clarity, and provide rationale for any scaling action that is occurring, especially on the x/z coordinate inputs.

################

In the generated flow diagram you note "generateNeighboringPoints3D" uses cached points if they are available, but in the function "generateNeighboringPoints3D", 


########################

The noise filter does run, but it takes significantly longer to run than any other sampler.  To improve execution speed, accuracy, and configurability, I have proposed these changes.  Alternatives may exist.
1. Instead of a parameter "frequency", use a parameter "gridsize" to derive how large the grids are to evaluate.  This will make it more clear to the user how it functions.
2. Add a parameter "branches" to point to another sampler that will be queried at the center of the cells location to determine the number of branches (In the code called "SUBDIVISIONS") to implement for each cell.  The number of branches should them be stored with other cell information, and should be stored in the cache so it does not need to be recalculated once the cell properties are calculated.  This will allow the number of branches to change on a cell by cell basis.
3. Update subdivideSegments (and other functions if necessary) to use CatmullRom splines instead of straight linear interpolation.  Add adjustable factors as necessary to give segments  more curvature, noting that more aggressive curvature is typically found at the lowest elevation (branch levels n=1).  Note this may require additional property data to be stored in the Segment3D, especially to get accurate distance calculations.
4. Update the return option "Distance" to return the 2d distance, if it is not already.  This may require updates to the "findNearestSegment" function.
5. Update the return option "Weighted" so it truly returns the minimum weighted 2d distance to any segment.  Noting this means the weighted distance to each segment would need to be computed in "findNearestSegment" and the minimum to be return, and therefore it would also require the segment to carry it's level property data, so it's weighted distance can be computed.
6. The point cache should NOT be initialized, instead the runtime will simply allow cell points at level 1 to be calculated inherently (by just not existing in the cache, they will be generated only when needed) and each cached point should have an age that allows the oldest cache point to be replaced by the newest query point if it does not exist in the cache.
7. It appears all math functions are being performed using default java functions, can seismic be utilized?  Does Paralithic have application?
8. There are multiple for loops that don't appear to have sequential dependency, can these be run in parallel?  Or will that already happen automatically?
9. The "distance" limit of generateSubSegments should likely be something more flexible than a hardcoded 10.0 limit.  Perhaps it should be based on grid size and number of segments to derive what the connective limits should be.

###################################

Note now the addon appears to get stuck while rendering, or it may be taking an incredibly long time to complete.  Implement a method to test the execution speed of the addon and implement build switches if applicable that might make it possible to test different compilable options and their affect on execution speed.

The benchmark ran successfully. Very interesting results:

Baseline (n=2): 3,090 samples/sec
No Cache: 2,713 samples/sec (-12.2%) - cache helps
No Parallel: 2,828 samples/sec (-8.5%) - parallel helps slightly
No Splines: 1,674 samples/sec (-45.8%) - WAIT, this is SLOWER without splines? That's unexpected!
Minimal: 3,445 samples/sec (+11.5%) - FASTEST!
High Res (n=3): 908 samples/sec (-70.6%) - as expected, more levels = slower
The "No Splines" result is surprising - it's actually SLOWER without splines. This might be because:

The linear subdivision is creating MORE segments than expected
Or there's a bug in how the linear vs spline path works
Wait, looking at the code, when useSplines=false, it uses subdivideLinear which still subdivides the segments the same number of times. The difference is just the interpolation method. But spline interpolation involves more math (Catmull-Rom calculations), so linear SHOULD be faster, not slower.

There might be a caching or warmup issue. Or the JIT compiler is optimizing the spline path better for some reason.

The key finding is:

Minimal (all optimizations OFF) is actually the FASTEST
High resolution (n=3) is 70% slower than n=2
This suggests the "optimizations" (cache, parallel, splines) are actually adding overhead rather than helping. For this scale of problem:

Caffeine cache has overhead
Parallel streams have thread pool overhead for small segment counts
The spline calculation might be getting JIT-optimized better than the simple loop
The user should test with:

use-cache: false

use-parallel: false

use-splines: false (or true, the difference is odd)

Keep n: 2 or lower

The benchmark works. Here are the key findings:

Benchmark Results
Configuration	Throughput	vs Baseline
Minimal (all OFF)	3,445/sec	+11.5% FASTER
Baseline (all ON)	3,090/sec	—
No Cache	2,713/sec	-12.2%
No Parallel	2,828/sec	-8.5%
No Splines	1,674/sec	-45.8%
High Res (n=3)	908/sec	-70.6%
Key Insights
The "optimizations" are actually adding overhead - The minimal config (cache=OFF, parallel=OFF, splines=OFF) is the fastest

Resolution level (n) is the biggest factor - Going from n=2 to n=3 drops throughput by 70%

Caffeine cache overhead - For this use pattern, the cache adds ~12% overhead

Parallel stream overhead - Thread pool coordination costs more than it saves for small segment counts

#####################################################

The issue now is that the root tree does not connect across cells.  It appears the key cell point q does not have any other segments to connect to, resulting in the entire segment chain staying isolated within a cell.

This means segments must be computed outside the cells, since each cell must have a guarantee of a segment point, neighbors must be known to determine segment connections.  Hopefully the random property is truly consistent for a single cell per salt / seed.  Initialization / call of random function may need to be investigated / verified.

Looking at why root segment does not traverse / span cell for interconnection:

The issue now is that cells are not connected to each-other with segments (SegmentsNotCrossingBoundaries.png, which spans x and z from about 0 to 4000 roughly), this appears to be due to how this filter code was implemented originally, as it seems it was only intended to emulate segment connections within a single cell region.

In order to rectify this, I would like you to implement the following changes, but there may be alternatives to ensure segments crossing the level 1 cell borders.

1. Root level segments (level 0 segments) must be connected between bordering level 1 cells so that their branching segments might influence the query cell.  To achieve this:
    A. A "resolution" level of 0 must be allowed (currently validator only allows resolutions down to 1).  0 should be the default resolution.
    B. At level "0" / step "0", select cells by returning all cells bordering around the focus cell.  This will create a grid of 5x5 level 1 cells each with a single level 0 key-point, each cell having a level 1 gridsize.  Note these cell key-points would ideally be located at the minimum value of the control function within each level 1 cell, but a rough minimum approximation is okay.
    C. Connect all level 0 points to create segments based on closest values as done with other segments (generateSegments), with the key uniqueness that level 0 points have an elevation of 0 and their connection should be based on closest 2d distance.
2. Trash / remove all segments not connected to a point in the inner 3x3 level 0 cell array, because without other cell context further away it is not possible to know where segments outside this zone should be connected or branch.
3. Continue branching / segmenting as done today for level 1, but instead of performing branching only in the query cell, also perform branching in the 8 surrounding cells.  This will be significantly more computationally expensive, but it will ensure any segments that grow can be connected between cells.  It is important that segment connection is checked for all segments across all cells (generateSubSegments)
5. Now we can remove any branch / segment that meets both of the criteria:
    A: Starts in a cell that is not in the query cell.
    B: Whose end point is moving away from the query cell.
    In this way, they would not influence (or shouldn't?) any branches / resolutions from cell 1.
6. Now continue branching, but again noting branching / segmenting must also occur in adjacent level 1 cells to ensure cross-cell segments can be created.



        # This generates the neighboring points... to query?  Or to generate segments from?  How does the tool know if the
        Point3D[][] points1 = generateNeighboringPoints3D(cell1, 9);
            This function drills in:
                private CellData getCellData(int cellX, int cellY) {
                    if (useCache && cellCache != null) {
                        return cellCache.get(packKey(cellX, cellY));
                    }
                    // Direct generation without caching
                    Point2D point = generatePoint(cellX, cellY);
                    int branches = computeBranchCount(cellX, cellY);



        List<Segment3D> segments1 = generateSegments(points1, 1);
        int branchCount = getBranchCountForCell(cell1);
        segments1 = subdivideSegments(segments1, branchCount, 1);
        displaceSegments(segments1, displacementLevel1, cell1);

I've updated the function to ensure two level 0 segments are connected so that there are not gaps between cells, but now some nodes are connected more than once through separate nodes.

Can you update "generateLevel0Segments" or create a new function that runs after level0 segment creation that ensures nodes are only linked through a single segment or set of segments (only a single path should exist from any level 0 node to another level 0 node)

Also, please confirm that the point data for each cell (CellData data = getCellData(cellX, cellY);) provides a consistent response for the X/Y coordinates even on successive calls, as in subsequent calls it appears to use rng.nextDouble() to generate new random values, but anytime a point is queries at a specific level for a specific position, it should return the same value.  It is not clear to me if that is occurring.

#####################################################33

"LinesAreStraight.png" shows level 0 lines are connected as expected, but the segments are very straight.  Is the distance actually returning the distance to the spline?  If so, what can be done to pronounce the curvature of the splines?

###########################################################

"SlightCurvatureButSegmented.png" shows there is now slight curvature, but some segments are once-again not connected.  When subdivision is occurring are some connections getting lost?  In regards to the lack of curvature, when displacement is occurring, is it weighted against the point distance or is it fixed?  Finally, most of the connections are either horizontal or vertical, due to the way the cell 0 points are positioned randomly with flat probability and the fact that closest distance is Euclidean.  Is there a way the point distribution into cells or distance calculation can be modified so that all connection angles have a similar probability of occurring?

###########################################################

"Disjointed.png" shows some disjointed behavior at the grid boundary, this picture was located at x=275,z=875, to x=650, z=1340.  Is the method used to segment and displace spline points also deterministic (derived from the connected points and salt), or could that randomness be causing these boundary artifacts?

###########################################################

I have reverted segment connections back to MST, since the branches are calculated locally, there will always be a risk of branches outside the focus cell to be connected more than once, this is okay as they will be very far apart.

There is still some discontinuity (Discontinuities.png), is there any thing else that would cause segments or their splines to not have the same definition when evaluating from different level 1 cells leading to these boundary discontinuities?  Any other random input that should be tied back to the level 0 segment definition instead of the cell?

#########################Hold

1. The fix you implemented around spacial hashes works but couldn't this be as simple as selecting the start point of each segment to spline as being the point with the lowest x/z values?


###########################################################

This method helped but there are still some discontinuities.


The MST does cause discontinuities as the network changes when the query cell is changed and thus is not tillable.  This problem is inherent using the MST method, but the method you implemented before as an alterative with cell anchors appears to create very unnatural looking connections and leaves some regions unconnected.  To make sure cells are able to tile without discontinuities and ensure all nodes are globally networked, while reducing duplicate branches between nodes, I have the following proposal:

1. At level 0, evaluate a 7x7 cell array instead of the current 5x5 cell array, so that interactions with cells (like duplicate chains) further away from the query cell can be rectified giving consistent influence for the query cell.  This should be tied back to a #define or config constant, called duplicate branch suppression.  For each integer number, the level 0 array should be expanded by 2, so a value of 1 (default) will result in a 7x7 cell array, a value of 2 will result in a 11x11 cell array, ect.
2. For all points, instead of using MST, connect each point to at least two other points, preferring points with lower or equal elevation via the control function, but if none are available link to the closest 2d point.
3. Now all nodes / points have been connected, for all nodes, if any 2 nodes are connected through a chain of segments with a length 2 segments or less (1+duplicate branch suppression), prefer the shortest chain.  If both chains are the same length, use a deterministic but pseudo random method to select a chain to keep and remove all others that have a segment length less than the duplicate check level (1+duplicate branch suppression)

###################################################################

1. Cell 1 points should not be random (generateLevel0Points should not use getCellData to place their points), instead it should use a rough approximation to get the lowest point (lowest control value) in the cell (this could be a very rough minimization technique to approximate the global minimum in the cell, ex: taking 25 random points and selecting the lowest)


###########################################################################

There should still be a little bit of pseudo randomness in the points queried for the minimum of the cell so there is not a grid formed across cells.  Add a little bit of deterministic jitter, the jitter applied for all query points can be the same for all points since only the minimum will be returned anyways.  (findLowestPointInCell)

############################################################################

Context for below: It's important the network achieves interconnected segments without orphan networks, while also being tillable, this requires each inner cell array (3x3) to guarantee it has a path out to the 5x5, but also that shifted evaluations result in the exact same shape for the query zone (3x3).  This means connections must be resolved to the 5x5 array?  Or how can this guarantee segmentation regardless of focus.  Does this mean every other cell needs an exit path?  How can that be guaranteed while also remaining stable?

New intent: Tile all 


Continuity appears to have been achieved. (ContinuityAchieved.png)

However, it appears some closed regions / orphaned regions still exist, as that should not be possible if all nodes require at least 2 connections.  Additionally, some duplicate chains appear to not be getting removed.  Are some duplicate chains getting removed that end up resulting in some nodes being orphaned?  This also should not be possible

#-----

Continuity appears to have been achieved. (ContinuityAchieved.png)

However, it appears some closed regions / orphaned regions still exist.

Rewrite the node / point connection strategy again for level 0 segments.  Remove / clean-up previous implementation.  These changes will exist in a branch to verify which method produces more naturally looking distributions knowing the limitations involved.

In order to both guarantee a global network (no orphaned points / nodes) and to minimize duplicate connections, the algorithm should add a level 0 cell than contains multiple level 1 cells, and defines a segment network for each level 0 cell the defines the segments to level 1 cells within.  The method to connect nodes within a level 0 cell would be as follows:
A. Connect all cell nodes by their closest distance (using euclidean distance, as the found points are already spaced according to defined minimum).
B. While any segments do not form a full tree structure to every node in the level 0 cell:
    i. Starting with the lowest node, if there are adjacent nodes that are not connected to this node, make a segment to that node.

To fulfil a query for a specific point.
1. Determine a cell level 0 position, where level 0 cells are k times larger than the gridsize, where k is configurable from 1 up to 10.
2. Determine which level 0 cells need to have their networks computed.  Since the query cell needs to have it's adjacent cells known, if the adjacent cells are located on different level 0 cells, those level 0 cells must also have their cell networks calculated.  This means if a query cell is located at the corner of it's level 0 cell, all 4 surrounding level 0 cells must have their segment networks calculated.
3. Compute the full level 0 segment network for all required level 0 cells.
4. Now connect the level 0 cells by stitching their adjacent sides together using the following mechanism:
    Per each adjacent level 0 cell:
        For each level 1 cell that is on the boundary of the level 0 cell, find the cell that has it's key-point at the lowest elevation, and create a segment between it and the adjacent level 1 cell in the other level 0 cell.
5. Now all adjacent level 1 cells are fully defined around the level 1 cell where the query originated from.  Remove all cells and segments that are not a part of the query cell or it's adjacent cells.

########################################################

I still see occasional discontinuities at cell boundaries.  Can you see anything else that may cause these discontinuities with the updated approach?  It is possible these were occurring before and I did not notice them.  Anything that could cause a segment to change / shift between two cell evaluations?  To me the discontinuities do not appear severe, I still suspect something in spline construction is being evaluated differently between two level 1 cells.

############################################################

The changes work, but why is subdivision and displacement affected by pruning?  Isn't subdivision and displacement per segment?  If some segments are removed, shouldn't the subdivision and displacement of all existing segments occur the same way?  Explain the interaction that is causing subdivision and displacement to be affected by neighboring segments.

###########################################################

I would like to take a different approach.  For each connected node of level 1 cells, we can determine the control points or tangent handles for spline creation separately from segmentation and displacement.  The angle and of any segments splined into the node can be up to 90 degrees from the linear connection, which will allows interesting curvatures around node points, the strength would need to be limited in a way to prevent splines from neighboring splines from overlapping, those could be exposed tuning parameters.

If a node is connected to more than 2 other points, the spline angle into additional paths should be similarly rotated to the main control points so that it's spline does not overlap with other splines entering the node.

This way control points for node connections can be defined first,  pruning can still occur before subdivision and displacement without affecting spline creation off of cell 1 nodes, and can create deterministic creation of splines.

##########################################################

I am seeing discontinuities, I suspect this is because cell level 0 stitching needs to occur when the query cell is both directly by a level 0 cell and at least 1 cell away from the level 0 boundary.  Update the logic for level 0 cell network creation to also trigger if the level 1 query cell is 1 cell away from the boundary (not just on the boundary).

Investigate if there are any other sources of discontinuities or if there is a separate root cause.

After creating segments and defining tangents, prune all cells that are not in the 3x3 level 1 cells centered on the query cell.

######################################################

From Claude:

I see another issue. Level 1 segments are being generated from already-pruned level 0 segments, which varies by query cell. Let me restructure the flow to:

Generate L0 for wider area (5x5 - already done)
Subdivide/displace L0 (unpruned)
Generate L1 from unpruned L0
Subdivide/displace L1
THEN prune both to 3x3

Continuity has been resolved but the tool is taking a very long time to load, I believe some new inefficiencies have been introduced because of the order of operations which I think are unnecessary.

What I would expect:

1. Determine what L0 cells need to be created to solve query cell (already updated based on level 0 cell interception with the 5x5 level 1 cells surrounding the query cell.)
2. Determine the level 1 segments to connect those level 1 cells bound within level 0 (Already done today, "generateLevel0Network")
3. Determine level 1 segments that stitch cells between level 0 cell boundaries. (Also appears to be done in "generateLevel0Network")

Determine the tangent definitions for each node / level 1 cells NOT on the border of the collection of L0 cells, since those nodes can't be solved because we don't know how those border cell nodes be stitched in with other cells in L0 cells that haven't been computed.
Now prune all segments that are not connected to a point in the 3x3 grid level 1 cell set surrounding the query cell.
Now perform subdivision and displacement on all segments, which at this point would include any segments both directly from cell level 0 and segments that were stitching at the border.  This way subdivision and displacement is only performed on the segments which must be solved for the query cell.

Please continue, but note we appear to have conflicting semantics which need to be aligned on.  I will restructure some of my definitions:

Level 1 cells are cells of the gridsize provided as an input configuration.
Level 0 cells are cells that contain fully networked level 1 cells.
Level 1 segments are those that connect level 1 cells.  Note there is no such thing as level 0 segments.  There are segments that connect the cells bound by the level 0 cell, but those segments are still connecting level 1 cells, and are thus level 1 segments.

With that clarification please proceed:



1. First, connect all cells in the 5x5 array to their adjacent cell / neighbor with the lowest elevation. (Only directly adjacent - top / bottom / left / right)
2. If any of the inner 3x3 nodes does not have a connection 
2. Now, perform a looping action to guarantee all points in the inner 3x3 array have a segment path outside the array (as this ensures they are are connected to the global network of points).  While any of the 9 nodes inside the 3x3 array do NOT have a path outside the 3x3 array:
A. S


At later levels (2/3/ ect) the points and segments should still be generated for surrounding level 1 cells so that their segments can be selected by the center cell level 1 points (to ensure points near the border of the center level 1 cell can be jointed / connected via segment to another point in an adjacent cell).  However adjacent cells only need to have points constructed / evaluated within their valid segment zones (don't try to generate / connect points where segments have been pruned)

Verify cell-crossing segments for other levels (level 1/2/ect) is also deterministic when building the spline.

If a point does not have a segment path back to level 0, it should be removed so it does not create orphaned artifacts.

After segments are split, are their nodes used both to create the spline and to act as nodes for higher level segments to connect to?

Implement an additional parameter that may improve query speed:

Create a new parameter:
cachepixels with a default of 0, scalable from 0 to gridsize.

If gridsize/cachepixels exceeds 65535 (UInt16 max) throw an error.

If cachepixels > gridsize, throw an error.

When enabled (any positive non-zero value) this will force the algorithm to cache the low resolution "pixel state" for each cell segment in "pixel" coordinates in the query cell for the defined resolution so the entire branch structure need be computed only once for a single cell.

A constant 20 MB max would be allocated to this cache.
This cache would be formed via a 2d array with dimensions of cell index / level 1 cell coordinates (x/z).
The amount of "pixels" allocated to each cache element would be (gridsize*(n+1)*3 / cachepixels).
If a single cell allocation exceeds the cache limit (20 MB), an error will be produced.
Else each "pixel" stores 9 bytes of information:
 A. An evaluated x,y position rounded to the nearest UInt16 integer offset from the cell origin in uint16, which represents a sample point the spline intersects.
 B. It's elevation stored in single of the spline.
 C. It's evaluated level, stored as a uint8. 

Thus, a 1000 size grid with a cachepixel level of 1 and query resolution of 2 would use 81 kb.

This would leave enough space for 250 cells, which would be organized so that the oldest queried cells would be removed to make space if a new cell needed to be queried.

To utilize the information from this cache, create two new return types:
pixel_elevation
pixel_level

If either of the new return types is requested, the algorithm can first see if the query cell is already cached before recalculating segment branches.

 If not cached, the "pixel" information would be calculated and stored after each segment is computed so long as cachepixels was enabled.  After calculating all the levels, or each level, or each segment, the segment would be evaluated along each cachepixels distance.  It's x,y position is rounded to the nearest integer offset in the cell according to it's cachepixels resolution, it's elevation is stored in single, and it's level is stored as a uint8.  If the allocated RAM for a segment is consumed, further processing for the cell could just be terminated prematurely, as this is a very rare scenario, and would at the worst just leave gaps / incomplete higher level definitions.

 Once the cell has been fully evaluated, the algorithm now has a cached definition of pixel information for the cell.

 If "pixel_level" is requested, the queries x/z samples just need to be rounded according to the pixel resolution the same the pixel was created from the segment, and if that pixel exists in the cell element in the cache, it's respective value (level or elevation) can be returned.

 ######################################################################################

 Update "DendryBenchmarkRunner.java" to make it easier to configure test cases and descriptions in a table-like format instead of their current spread form, and add a test case to compare the "cachepixels" flag option compared to baseline.

 ########################################################

I need your help understanding why the most recent updates don't improve performance.

I updated the benchmark with test 7 ("7. CachePixels Enabled (NEW TEST)") to evaluate the pixel cache method with a return type of pixel level.

I expected cache pixels with pixel value to be much faster because the tree is not rebuilt for every sample, and distance to spline does not need to be calculated for every sample.  Instead a single longer first iteration would run to compute the pixels along the segment, afterward subsequent calls should just be looking up values from the pixel map.  Given distance computations should be similar to just populating a pixel, and we are only computing along the segments (not all the empty space around the segment) I would have expected the timing to decrease by about the square root of the previous tests that report distance.

Can you:
1. Verify I setup the new test case correctly.
2. If correct, add more debugs to root cause if there is an issue (ex: Set up a counter to see how many times a network is calculated during a test, to understand if something is triggering network / segment rebuild superfluously.
3. See if any implementation / test methods are obscuring results.  Ex: Is the test scenario being used causing the filter to reinitialize for every sample instead of allowing it to take advantage of cached information.

############################################################3

Level 0 segments (created both inside cell level 0 and as stitching segments) have tight / non-continuous joints and repetitive structuring.  To fix this:
    A.  When segments join at a node, at least two of the joining segments should have tangents that are opposite angles to give continuity along the node.
    B.  When creating segments between nodes based on distance, the shortest distance should be compensated for the square grid structure (Should already be implemented at higher levels - to break up the grid-line appearance.)

#########################################################################

I have reverted the grid compensation for now due to overlap, try to use distanceSquaredTo for the level 0 segment distance calculation and I will review the results.

I still see severely jagged connections where only two level 0 segments connect.  Is it possible the pixel value is getting allocated to it's position along the linear interpolation between the two points of a subdivided segment instead of what should be a smoother spline?

Check pixel creation
Check tangent creation

#########################################################################

Apply the following changes to this project:

1. Raname Level 0 segments to be called AsterismSegments
2. Rename Level 0 cells to be called constellations.
3. Rename Level 0 points / key-points to be called stars
4. Rename Level 0 networks to be called Asterisms.
5. Function "computeAllSegmentsForCell" duplicates much of the logic in "evaluate", can these be merged in such a way that logic trees are not duplicated in different functions?

6. ConstellationScale should replace level0scale.  This should be a value greater than 1.

#####################################################################

5. The "branchesSampler" parameter can stay as it will still be used in future clarification, but all other references to "branchCount" should instead be called "segmentdivisioncount" or something similar, since "branchCount" is not actually creating branches, it's just dividing the segment into pieces for spline creation.

Refactor this code to:

Function Cleanup:


Please make the following changes with the limited information.  Placeholders may be needed until I provide additional information.

Note the definition of a segment changes to contain additional information (defined in CleanAndNetworkPoints)

Note "level0scale" has been renamed to "ConstellationScale"
Code changes:

* Have a hard-coded parameter describing "merge point spacing" with a value of 2/3 of a cell (This parameter will be used later at multiple levels with differenct cell sizes).
* Have a hard-coded parameter describing "maximum point segment distance" with a value of sqrt(8)+1/3, as this is the maximum size two adjacent stars can be located after potential merging (This parameter will be used later at multiple levels with differenct cell sizes).


* Have an additional configuration parameter to define the tileable shape for  constellations, including Square / Perfect Hexagon / Rhombus, default Square.

* Constellations scale with the parameter ConstellationScale, where 1 scales the constellation such that the largest possible inscribed square (without tilting) would be 3 gridspaces wide.  (This ensures when the constellation is tiled, only the 4 closest constellations need to be solved to resolve the local network of points around the query cell)

* When a cell is queried, if pixel cache is used and available, return the data from the pixelcache as is done today.

* Else proceed to solve the network to solve the queried cell:

* Determine the four closest constellations to the queried cell based on the query cell position and the constellation locations (derivable assuming a starting constellation position centered at 0,0 coordinate)

* Iterate through each of the four closest constellations to define the network within the constellation:
** Determine all the level 1 cells needed to circumscribe the constellation.
** For each level 1 cells circumscribing the constellation:
*** Perform sampling within each cell via a 9x9 offset grid of potential stars (this should be deterministic), select the star with the lowest position using the control function, if there is more than one lowest point, randomly select one.
*** Note now all stars have been drafted in the cells circumscribing the constellation.
*** Remove any drafted stars that are outside the boundary of the constellation, or within 1/2 of the merge point spacing from the boundary.
*** Now go through all stars while any distance between any star is less than the star spacing, and merge those stars into a single star.
*** Note now all stars have been set in the constellation.
*** Now perform network creation within the constellation to define the Asterism.  The method to create a network of points (CleanAndNetworkPoints)  will remain consistent per level, with some special rules for Asterisms (level 0).

New function: CleanAndNetworkPoints
Inputs:
Unique cell location definition (x coordinate, y coordinate, level)
Outputs:
    Segment definitions (to be added into other segments)
    Each segment has two 3d points (x,y,z), and two tangents describing the end condition in the x,z coordinates.

Instructions for stitching the asterisms together and the exact method to network points will be described in a future point.

###############################################################################


PH: 
B. Then if there are remaining points, iterate on each lowest point using the connection rules below, performing connections to 

#####################################################

New behavior / properties:
    - When points are created for a cell (generateNeighboringPoints3D) use their closest 2 neighbors that are at least 30 degrees apart to estimate the x and z slope (stored as properties in the 3d point).
    - Add two more methods to 3d points to get:
        - Tangent - Calculated as angle of the gradient normalizing the x,z slopes.
        - Slope - Calculated using the x and z slope components to get the slope along the tangent.

New parameters (Placeholders which will be referred to in future):
    - SlopeWhenStraight: Defines when points will have their tangents fully aligned with with the slope without any variance, from 0 to 1, default 0.1
    - LowestSlopeCutoff: Defines when points must be rejected, typically a positive number to indicate a positive / upward grade of flow which is not reasonable, but if still greater than 0, lower level segments will be reduced in elevation to ensure the branches do not "flow" upward.  default: 0.01

################################################################################
Note "NetworkPoints" has been renamed to "CleanAndNetworkPoints"

Note: "Flow path" is used below but is effectively synonymous with "tangent", but for consistency the segments should be populated such that point a is the start of flow and point b is the end of flow.



Now the asterism is created using this function.  The pixel level return type can be evaluated by calculating the spline according to it's tangent / point combination, no longer limited to linear interpolation.  Only the pixel return type need be updated, as the other return types would likely be extremely slow given distance computations on splines.  The pixel elevation will be updated later. Higher levels (1+) can be updated later to use "CleanAndNetworkPoints" after initial evaluations and assessment.            

###########################################################

For stitchConstellationsNew:
Inputs:
    The segment definitions of the two asterisms being joined.
Outputs:
    Segment(s) definition describing how two constellations are connected.

Find the two points in the two asterism definitions within the maximum segment distance with the smallest absolute slope between the two points.
For each point in the new stitching segment:
    If the point it's connecting to in the asterism is the end of the line, set the tangent equal to the tangent of it's connected line so it's continuous.
    Else if the point it's connecting to is already a part of a line, set the tangent to a random but deterministic angle between 20 degrees to 80 degrees from the continuous line.

########################################################################

In Segment3D and Segment2D, rename point "a" to "srt" and "b" to "end" to give context to the flow direction, this will be clarified in the future to drive consistency in "CleanAndNetworkPoints".

Note I've commented out computeNodeTangents, subdivideSegments, and displaceSegmentsWithSplit in generateAllSegments, as those steps will be achieved through "CleanAndNetworkPoints", and will be removed completely once new implementation is confirmed.

########################################################################

There are multiple discontinuities in the new implementation as well as crossed segments, it appears this is is due to how "CleanAndNetworkPoints" is implemented from steps 4 through 8.  Right now those steps are implemented as large sequences per point / star group, but the sequence of steps should be performed each time a segment is created (when a point is connected to a neighbor) to prevent overlap and duplicate node connections.

Review "NetworkingRules.md" and refactor steps 4 through 8 in "CleanAndNetworkPoints" to perform functionality on a per segment creation basis.

#############################################################################

When subdivisions are created, I would expect the new node to break the parent segment and become the connection point for two new nodes.  That would have also prevented infinite looping.  Please ensure when subdivisions are created they are also replacing the devided segment with two new segments, such that unconnected points are not left "floating"

###########################################################################

A few notes, currently only looking at level 0 / asterisms:

1. Some segments still appear to overlap, is there something else in the implementation that may allow segments to overlap?  Or are there some non-deterministic randomness being used which is causing segments to change paths depending on which x/z point is being evaluated?
2. Some segments still appear to be orphans (they don't connect with the rest of the group), this may have been due to some ambiguity in NetworkingRules.md, which I've tried to clarify on line 23.  I've also removed the maxIterations for level 0 because all level 0 points should connect.

########################################################

I am still seeing many crossing and orphaned segments.  Please implement some additional functionality so I can more easily debug, though suggestions are welcome.

1. Add a new return type "PIXEL_DEBUG" that returns 3 when the sampled point is within 2 pixel spaces (cachepixels) of an original star / point segment.  Returns 2 when the sampled point is within 2 pixel spaces of a point that was created as a part of segmentation.  Else returns 1 when the sampled point is on a segment (similar to level, but always 1).  Else if none of those conditions match return the same value for level as is done today when there is nothing (I believe -2 or -1).  This will help me visualize where different points are getting created.

Also ensure setPixel is not able to set a pixel value more than once, as that may be consuming the cache with duplicate points if protection does not already exist.

#######################################################################

I am seeing multiple segments that don't appear to end with an original point segment and I don't see any points from segmentation.  It should not be possible to have a segment that does NOT have end-points that are either a star / point or from segmentation.

Implement a hard-coded parameter that returns segments at different levels before they undergo further processing so I can change the parameter, recompile, and verify at what step where some of these issues start to appear at?

Something like "SegmentDebugging" where:
0 - Normal operation
10 - Return the segments for the first constellations before stitching, tied to their closest neighbor.
20 - Return the segments for all the constellations before stitching, tied to their closest neighbor.
30 - Return the segments for all the constellations including stitching, tied to their closest neighbor.
40 - Return the segments up to the first call of phase A in CleanAndNetworkPoints
50 - Return the segments up to the first call of phase B in CleanAndNetworkPoints



##############################################################################

Add SEGMENT_DEBUGGING = 40, where the segments are constructed from the boundary of the Constellation, is the merging methodology 

###############################################################################

It appears that ConstellationScale is not actually resulting in the constellations growing in size.  As I change ConstellationScale between 1 and 10, the segment outcome appears to be the same, can you investigate why ConstellationScale does not appear to be expanding the outcome constellations to have more stars / occupied with more level 1 cells.

###################################################################################

Review the current subdivision functions used in connectAndDefineSegments, revise this or make a new function to ensure the following behavior, and indicate the changed / new function(s) that accomplish this.  Make sure after running this that the original segment that was subdivided is properly removed if not already done:

A. When a segment is subdivided, it should be implemented in a way that it can be subdivided into x number of segments (function input with number of divisions) in case this needs to be changed / tweaked in the future. The allowed jitter factor should also be a number from 0 to 1 indicating how much jitter / displacement is allowed for subdivided points, which may need to be different at different levels to reduce complex shape generation.
B. The division of the segment should occur along the b-spline interpolation of the segment when the tangent information is available.  Largest jitter / displacement of the segmented point in the x/y plane should be limited to the original segment point to point distance divided by (number of divisions *2*allowed jitter factor)
C. Any new points (knots) created by subdivision should have a point type of "Knot", to help differentiate from their initial "point" types.
D. Any knots created by subdivision should inherit their tangents from the b-splines that created them.

Make sure the Endpoint type flags for debug visualization is updated to an enum / integer instead of a boolean to since the knots can only have a definition of: 
- Original (point was originally created)
- Trunk (will be described later, a point that was a part of the original trunk creation)
- Knot (point was created as a part of subdivision)
- Leaf (will be described later, a point that exists on it's level that is the end of a branch)

########################################################################

After reviewing the output, segments are not connecting / branching as I would expect.  I suspect the points created as part of segment creation and subdivision are not being used when potential neighbors are being found for subsequent point connection.  I have attempted some clarifications in "NetworkingRules/md", please review this updated text, plan the implementation, ask clarifying questions if necessary, and implement the changes when ready.

###################################################

Reviewing the output, it appears the tree (which should be a single unbranched segment) does have branches.  Can you check to see:
1. if some other process is adding segment branches when SEGMENT_DEBUGGING is set 15 after tree creation.
2. If there is an issue in "PIXEL_DEBUG" that might result in additional segments appearing to get created.

####################################################

Now add a hard-coded option in the method / function which populates pixes for the PIXEL_DEBUG return method to populate the pixels based on the evaluated b-spline using the segment tangent information, instead of what appears to be current linear interpolation.

#######################################################

I see discontinuities with some segments which I believe is due to tangent creation.  I have updated "NetworkingRules.md" with lines 68/69 to help verify tangent behavior in some cases to ensure continuous segments (those that do not branch) have continuous curves.  Please verify the tangents are inverted when two "end" points meet according to this logic.

Please also add a hard-coded flag to allow the routine to terminate with an error if any constellation segments are returned with an undefined tangent.

########################################
I zoomed in and I see what appeared to be discontinuities are actually overlapping lines that tightly looped near the node.  Can you investigate why the tangent appears to have a vector that is moving against the segment flow direction?  The cross-product of the segment vector onto the slope vector should still yield a vector moving away from the start point, and even with the random twist applied I would not expect the deviation I saw (which appeared to be about 170 degrees)

I have added line 79 to "NetworkingRules.md" to attempt to bound tangents that point against the flow path, but it is not clear to me how the tangents are flowing backwards at all.  Is the tangent calculated for the point before it is adopted by segments normalized?

###################################

Additional changes to make:

1. Number of subdivisions in subdivideAndAddPoints (using "divisions" variable) should instead be calculated from the segment length (can just use start to end point distance), such that divisions = floor(segmentLength/"merge point spacing"), to ensure multiple nodes available on subsequent segment creation at lower levels.
2. Curvatures look very minimal.  Set tangent magnitudes to scale with "merge point spacing" when tangents are assigned along with scaler constant so that curvature is more pronounced (this assumes segments are getting created / interpreted as hermite splines)


################################

stitchConstellationsNew needs some fixes.  Please refactor it for updated behavior:

1. The constellation will only have stitches created for their adjacent connections.  This means a for loop will only go through the constellations a single time, each time a constellation should be stitched into the next constellation.
2. Up to the 6 pairs closest pairs between the two constellations will be evaluated.
3. The pair with the lowest max elevation between the two points will be selected to stitch the two continents together.
4. For each end of the stitch line:
    A. If the connection to the constellation is only connected to a single other point (the connection is a leaf) the tangent will be set so that the two segments are continuous.
    B. If the connection to the constellation is connected to two other points, the tangent can be set to match either related tangents.
    C. Tangent strength should use an identical methodology to that used in createAndDefineSegment.
5. After stitch creation, the segment should be subdivided using the same methodology as createAndDefineSegment, using subdivideAndAddPoints.

###################################

0. Wait till after constellation stitching is complete to force the elevation of segments / points to 0 so elevation is available in stitching decisions.  Currently elevation is forced to 0 as a part of constellation segment creation.

There are still issues in stitching occurring:
1. The order of the constellations being presented is creating an "x" pattern.  Make sure the constellations are ordered or the original array is initialized such that the constellation order creates a loop around the constellations (like clockwise or counterclockwise) so crossing segments do not occur.
2. When stitches are made, their tangents both appear less significant and their subdivisions smaller than constellation subdivision.  Segments should use identical tangent magnification and subdivision (at level 1) as constellation segments.
3. There are stitch segments that appear to no end in connections, and instead appear to be floating.  Stitching should NOT require the input "constellationStars" (for stitchConstellationsNew) as stitching should only apply to the already computed segments for the constellation in "constellationSegments"

#################################

Issues:

The constellation boundaries (empty space between asterism stars) appear to be much bigger than expected.  With a grid spacing of 1000 and a merge distance of 2/3, stars from different asterisms should get within 700 of eachother.  Current gaps are consistently 2000, which also indicates inappropriate bounding / removal of star points when creating constellations.  Please investigate this issue.  Considerations:
- Are the level 1 cells circumscribing the constellation proper?  The outside boundary of all level 1  cells should completely surround the constellation boundaries before cleaving stars outside / close to the boundary of the constellation.
- Are some calculations attempting to derive star merge distance at level 0?  Since stars are created at level 1 cells, their merge distance should always be for the level 1 grid size.

##############################################################################3

On line 894, please add the constellation size to the logger info.  There are significantly less bounded stars than expected, I believe this is either a size calculation issue or their circumscribed draft stars are getting created offset from the constellation boundary resulting in many of them being cut during bounding.

###############################################################################3

mergeCloseStars is implemented incorrectly, and should use the same method as used in cleanAndMergePoints.  Update mergeCloseStars and if necessary cleanAndMergePoints to use the same function to merge stars.  Merging them in such a way the maximizes the number of remaining stars but such that no star is within the merge distance from one another.

#############################################################################

In the logger on line 895 add the center of the constellation coordinates and lowest x, lowest y, highest x, and highest y of the stars before they were cleaved and merged.

###################################################

In the logger now on line 907, also add the boundaries after merging similar to the boundaries that are shown before cleaving.

1. Tangents for stitches are occasionally flipped from the orientation they should have, resulting in sharp / spike like corners instead of continuous / smooth curves. Verify tangents are getting inverted / flipped when necessary when on the ends of branches (leaves)
2. As noted earlier, the subdivision rules should be identical to constellation creation.  The subdivisions currently appear about 3x bigger than expected, which coincides to level 0 cells being referenced.  Since stitches are stitching level 1 cells across constellations, their segment construction and subdivision should be handled identically to segments that are created for asterisms.
3. Stitches appear to abruptly end sometimes at cell boundaries.  This indicates for some query cells, the wrong constellations are being solved such that a stitch ends up missing when that stitch is evaluated.  Are the closest constellations being returned based on the level 1 query cell to ensure close connections including stitches can be solved?

###########################################################

I have reverted previous changes as segments now are not appearing at all.

There are fundamental issues in the way constellation coordinates are handled. findFourClosestConstellations treats the constellations like square cells, but rhombuses and hexagons will not fall on this grid pattern.  A constellation should not be assumed as a grid.  Instead, according to the pattern layout of the shape, the closest constellation centers should be able to be found based on the center of the query cell.  Constellations need only be reported as their center coordinates, the start coordinate of the level 1 cells that circumscribe that constellation, and then number of level 1 cells horizontally and vertically to completely tile across / circumscribe the constellation shape.

1. Revise "findFourClosestConstellations", finding the 4 constellations that are closest should be sufficient for all shapes to connect their neighbors.

#####################################################

Constellation indexes should not exist at all.

Ex:
int baseConstX = baseIndices[0];
int baseConstY = baseIndices[1];

Update findClosestConstellations again, get the center of the constellation for the current query point (queryCenterX,queryCenterY), find other constellations by applying thn corresponding x / y offsets according to the shape to the initial center constellation.  generateConstellationStarsNew should use startCell and cellCount to form the draft star points.

##############################################

Refactor subdivideStitchSegment and subdivideAndAddPoints to use common subfunction(s) that performs the segment subdivision and adds jitter to the points so code is not duplicated between the two.

##############################################

To fix the tangents on stitches (constructed within stitchConstellations), make sure tangents are properly aligned for connections with leafs

For each end of the stitch segment that ends on the leaf of a segment:
If the stitch point type matches the leaf type (both types are end or both types are start), use the tangent from the leaf point but flip / invert it.
Else if the stitch point type is opposite of the leaf type (end to start connection type), just take the tangent from the connected leaf directly.

This should ensure continuity within the segment.

##########################################################

Increase constellation query for borders?  Issue with ordering and angle assignment?  Angle should be around const center?

S

There are cases where 6 constellations are necessary,  I have updated the number of returned constellations to 6.

Rewrite the method to select constellations to stitch together (function "stitchConstellations")

Instead of stitching each constellation to it's next constellation (the current for loop), only stitch pairs of constellations that have a shared border.  This can be done by finding pairs of constellations whose center-to-center distance is below the adjacent threshold, which can be calculated per shape, and is just the distance between the two adjacent shapes with a little extra for rounding.

Then loop through these pairs and compute the segments as is done today, but note the segment should be computed in a deterministic way if any randomness is performed, if not already done.

###############################################

getAdjacentConstellationThreshold is wrong - debugging

till issues with merging and bounding?

Do:

Update "isInsideConstellationBoundary" to cleave / remove points to the exact shape boundaries (not estimated using distance from the constellation center, and remove boundary checks).

Implement new Merge Rules laid out starting in line "NetworkingRules.md" at line 83

#############################################################

Skip merging entirely so that root is available at all cells.

Make sure segment connections forcibly connect any nodes within the merge distance.

Now update cell layers and compute for layers?  This should allow for full locality, additionally, pruning can be done fully to a single cell level?

###############################################3

I have made some minor update to "NetworkingRules.md" on the following lines:

- Line 63 to 66, this may affect "connectAndDefineSegments" and others who determine which point to link to, as this forces points within merge distance to be connected to with preference, this will inherently only affect level 0


1. Fix grid spacing definition for consistency.  There are multiple references to grid-spacing and level-related spacing:
    The spacing of lower-level segments should be defined by a number of divisions per level, like: [3,2,2] indicating there will be 3 levels of division on the first level, 2 on the 2nd, and 2 on the 3rd.
    Then points spanned per cell per level would be calculated as the multiple of all upper levels, such that would result from the example above as:
    [3,6,12]
    This means all gridSpacing should be divided by this points spanned per cell instead of the current power function: "double gridSpacing = 1.0 / Math.pow(2, (level+1))" 
    This would be use to compute the proper merge spacing and segment distance per level.

2. Now update the main loop to generalize higher level segment creation:
    A. Instead of stepping through each level through duplicate code (ex: "// Level 2+: Higher resolution refinement"), implement a for or while loop to go through and refine each level and return the full segment definition if the number of segments requested is reached.
    B. At each level create points (via generateNeighboringPoints3D) with the number of points per cell per level, only on the query cell.
    C. Use "CleanAndNetworkPoints" to create segments between the new points and the lower segment level.
    D. Add additional rules near "probabilisticallyRemovePoints" to start removing points randomly as the points get further away from the segments 1 level below in order to reduce branches / higher level segments from creating an obvious square pattern as they approach the edge of the cell.

################################################################

Are stitches attempting to connect to any segment node?  Or only stars / leafs?  They should evaluate for any node on the segment, not just stars.

Check segment distance for both star segments and for stitch <- This appears to be fixed.

####################################################

IMPORTANT:

Still reaching max iterations on chains? ==>

Level 0: connectChainsToRoot reached max iterations (630)

There is a defect somewhere.  Even at level 2 this should be impossible.

2 questions:

In "DendrySampler.java" why is level 1 in "generateAllSegments" not in the for loop?  Why is the for loop only for level 2+ (line 464)?

What could be causing connectChainsToRoot to reach max iterations?

######################################################################

Make the following updates:

1. Update the variable "asterismPruned" so it only contains segments that have at least one end within the query cell (level 1 cell size), and that segments that do cross the cell boundary are cut at the boundary and given a new knot / point type that indicates it is a EDGE point (public enum PointType). (DendrySampler.java line 439).

2. Update connectAndDefineSegments and it's related functions so that functions do not attempt to connect to EDGE type segment ends.

###################################################################

There appears to be a defect in "connectChainsToRoot" causing multiple overlapping segments.  Review NetworkingRules.md part B (starting at line 32) with additional clarifications / augmentations.

IMPORTANT: findRootChain (if it continues to be used after these updates / fixes are applied) may be overly complicated, a path to the interconnected root just means the node needs to have a connection to either a trunk type node or a node that was created 1 level down.

####################################################################

At level 1 I do not see any segments that return back to the main asterism / points connected to trunk nodes.  The level 1 nodes only connect to each-other.  Is there a bug preventing those segments that connect level 1 nodes to their level 0 parent segments from getting added to the pool or allowing them to be displayed?

##################

Did that bug also affect the first time connectChainsToRoot is called at level 0 after trunk creation, such that trunk nodes / segments weren't available to connect to?

#################

Keep stitches away from each-other.

FUTURE:

I would like to make some significant refactoring.  Below I describe core data structures to change with the expectation the NetworkNode type can be removed in it's entirety.

I have also updated "NetworkingRules.md" to reference the modified structure proposal below.

What to do: Remove NetworkNode in it's current form and refactor as necessary to meet updated descriptions below and in "NetworkingRules.md".

Update data information to contain context from NetworkNode:
    Node / Point3D (or something that includes Point3D):
        Connections - Add a new integer that contains the number of connections to this point.  Initialized to 0.
        Index - A key value that other methods or functions can find this point by.  Every newly created point should have a new index.  UInt32 should be sufficient.  The runtime can produce an error if the index exceeds the UInt32 value, as this should never occur.  Index should reset to 0 upon a new evaluate query.
        PointType - Copy from NetworkNode definition.
        Level - Indicates the level this point was created at.

    Segment3D
        The srt and end points should just be an index of a 3d point in a list of 3d points.
        Remove srtType and endType, as that will be a property of the point data.


New structures / packages of data:
    SegmentList: Intended to contain a list of interconnected / chained segments.
        Contains:
            List of points (Node / Point3D from above)
            List of segments (Segment3D)

    UnconnectedPoints (for new point clouds):
        Contains:
            List of Points (Node / Point3D from above) generated for cell level that have not been linked yet.

Looking at the NetworkNode type:

        Point3D point; - This is already carried by the 3dpoint information or is in the segment information.
        int index;     - This can now be handled by the point index.
        final List<Integer> connections; - This can now be handled by point connections
        Vec2D tangent; - This is duplicate, the 3dpoints have tangent / slope information on their points for the control function, and the segments have tangent information on their ends to form splines.
        boolean isBranchPoint; - Unnecessary, if a node tries to make a connection to another node that already has 2 connections, it must be a branch.
        int branchIntoNode;  - Unnecessary.
        boolean removed;  - Unnecessary, if a point is not used for segment creation, it just shouldn't be added to the segment chain structure.
        int chainId;  - Unnecessary, networking rules have been updated to always grow from a defined segment, either a point is able to attach, or it isn't.
        boolean isSubdivisionPoint; - This information is already contained in pointType
        int sourceLevel;  - Level of segment that created this point (for subdivision points)
        PointType pointType; - This is now handled by the point type.


Now any function that performs networking / linking activity is taking points from the new point cloud and directly tying them into the segment list structure, moving the points from the new point cloud to their new location in the segment list structure.  A separate "NetworkNodes" variable becomes unnecessary as node context is either apart of the segment definition, or it just hasn't been added yet.

########################################################

Traversal tracking will not be necessary.  A single segment list structure carries all linked / connected segments and new points are only added to that base segment structure.  That is - the base segment grows from a single tree - so independent chain creation, tracking, and integration is unnecessary.

The initial base segment list structure is created by merging all the the constellations and their stitching segments before proceeding to level 1+ segment creation.


###################################3

Post refactor notes:

1. Tree segments do not have curvature, are not splined with the right segment subdivision.
    A. When creating cell points, maybe give boundary?  Or make sure stitches happen along the center 80% of the boundary?
2. Branching is causing substantial cross-over, and doesn't appear to be targeting closest nodes at all?

#########################################

Is "buildTrunkV2" necessary in it's segment complexity?

Can findBestTrunkNeighborV2 be converted to findBestNeighborV2?  Just add 0 length segment to lowest point on trunk before running subsequent run?

Note, merge these into common function: 

    private int findBestNeighborV2(NetworkPoint sourcePt, SegmentList segList,double maxDistSq, double mergeDistSq, int level) 


    private int findBestTrunkNeighborV2(UnconnectedPoints unconnected, SegmentList segList, int currentSegListIdx, double maxDistSq, int level)

Add function variants (SegmentList.java - addSegment):

public void addSegment(NetworkPoint srtNetPoint, int endIdx, int level, double MinimumSegmentLength)
    Adds the start point to the list, uses it's index to add the segment.

public void addSegment(NetworkPoint srtNetPoint, NetworkPoint endNetPoint, int level, double MinimumSegmentLength)
    Adds the end point to the list, uses it's index to call the previous method

public void addSegment(int srtIdx, int endIdx, int level, double MinimumSegmentLength)
    Updated to compute tangent, maybe this needs twist magnification element.


Then createSegmentV2 can possibly be removed entirely?  Or does it call the addSegment method?
createSegmentV2(UnconnectedPoints unconnected, int unconnIdx, SegmentList segList, int neighborIdx, int level, int cellX, int cellY)

########################################################333

Update "addSegment(NetworkPoint srtNetPnt, int endIdx, int level)" in "SegmentList.java" to add content to the function and add inputs if needed, specifically so that it creates multiple connected segments from a single call to this function.

Follow the placeholder comments in the function.  More computation details can be derived from various functions in DendrySamplerOld.java:
    - Tangent computation should reference computeNodeTangent
    - Spline creation can reference connectAndDefineSegments
    - tangent computation, jitter application can also reference subdivideAndAddPoints 

- Compute that tangent as a property of the of the point information

####################################################################

Now update DendrySampler.java to use the new addSegment methods from SegmentList.java.

Trunk creation should create the first segment by calling the new addSegment method with two new points, then call the new addSegment method with a new point and the previous index to create the rest of the trunk segments.

Branch creation should create each growing segment by calling addSegment method with a new point and the index found existing in the segment list.

Asterism stitching should be created by call addSegment method using the two point indexes for the points that are going to be combined.

Subsequent level creation should use a similar path to branch creation.

###########################################################

In SEGMENT_DEBUGGING set to 15, I am not seeing the tree segments or any others normal segments appear.  I do see stars, which implies 0-length segments are getting created and rendered, but it seems something is preventing longer / typical segments from getting rendered or created.

##############################################################

Issues:

1. Discontinuities are present on trunk, something is causing segments to not get created deterministically.
2. The trunk does not continue upward from every newly created point, instead it is branching out, which is not expected behavior (checked with SEGMENT_DEBUGGING=15).  The trunk should be non-branching, is the newly created index point used as the connection point for the next iteration?
3. combineConstellationSegmentLists appears very complicated.  Please make sure new constellation points get their index continued from the last created point from the previous index, that way the points and segments do not need to be reindexed, and should be able to be copied over directly.

#########################################################3

Make changes if necessary so that addSegmentWithDivisions only applies jitter to new intermediate points, not to the start / end points used to start the creation of the segment.

##############################################################

Update getExistingConnectionDirection to actually find the tangent of connecting segment to the point, instead of the slope tangent of the point.

#################################################

Some subdivided nodes appear much closer tto the original point than I would expect.  Update or confirm the following:

1. That divided segment points are placed on the hermite spline equidistant according to the number of divisions to create, and jitter is added to those positions.
2. That tangent strengths are based / multiplied by the distances between the original points before subdivision.

################################################################################


Make changes if necessary so that the tangent derivation in computePointTangent at (point.connections == 1) is properly compensating the tangent for the new segment for continuous flow / derivative along the full segment chain.  If connected point is an end point and it's connected to the end point of the segment being created, or if the connected point is an start point and it's connected to the start point of the segment being created, the tangent must be negated / rotated 180 degrees so that the constructed splines align.  Otherwise the tangent should be taken directly.

############################################################################

Make changes if necessary to ensure points are appropriately categorized:

KNOT - Points created during subdivision that were not for trunk segments.

TRUNK - Points created during subdivision that were for trunk segments.

LEAF - Points that only have 1 connection after all segments have been created for a specific level or constellation.

ORIGINAL - The original points or stars created as a starting point to be connected through subdivided splines,

EDGE - Points that were created as a result of clipping a segment at a cell boundary.

####################################################3

There are many places where Segment3D is being used where SegmentList should be used instead so that the number of connections is easily detectable per point and to reduce duplicate point definitions between segments.

1. Remove the information which will not be necessary from Segment3D
public final PointType srtType;
public final PointType endType;
public final Point3D srt;
public final Point3D end;

2. In "generateAllSegments" update the following functions and their assigned variables to return SegmentList types instead of Segment3D:

generateAllSegments
generateAsterism
pruneSegmentsToCell

3. In "evaluate" the returned SegmentList from generateAllSegments can be converted to Segment2D for compatibility with computeResult, computeResult can be updated to only use Segment2D.

4. In "computeAllSegmentsForCell" it should also return a SegmentList.

5. "sampleSegmentsToPixelCache" should be updated to use SegmentList inputs, populating pixels by each segment solved using their tangents and end point indexes, followed by marking points according to their type.

Note "allSegments" that is returned from "generateAllSegments" should not occur until all levels are complete to ensure the correct 

Changes for this should simplify the code-base, if changes are resulting in additional or more complex code, it is likely something is being misinterpreted.

######################################################

I'm getting the following error:

java.lang.IndexOutOfBoundsException: Index -1 out of bounds for length 128
	at java.base/jdk.internal.util.Preconditions.outOfBounds(Preconditions.java:100)
	at java.base/jdk.internal.util.Preconditions.outOfBoundsCheckIndex(Preconditions.java:106)
	at java.base/jdk.internal.util.Preconditions.checkIndex(Preconditions.java:302)
	at java.base/java.util.Objects.checkIndex(Objects.java:365)
	at java.base/java.util.ArrayList.get(ArrayList.java:428)
	at dendryterra.SegmentList.getPoint(SegmentList.java:113)


    ############################################

createSubdividedSegments (line 466) needs some fixes:

1. Tangents should be created for each subdivision instead of being set to null, with some slight jitter / randomized twist (likely +/- 10 degrees.)  The amount of randomized twist should reduce as a function of how much jitter offset the point in the x/y coordinates.  Add the maximum amount of intermediate twist as a config in SegmentListConfig.java.

2. I still see some segments generate with a point very close to the original point, could there be some rounding or other computation error that would lead to an intermediate point getting created very close to the start or end point?

Some of the tangents appear to huge angle deviations causing significant twist.

####################################################

In SEGMENT_DEBUG 15, it seems there are values of 5 returned (Edge type points) but they are not on the cell / grid boundary.  Can you verify pruneSegmentsToCell is dividing only crossing segments that cross the cell boundary and are divided at the cell boundary?  Note segments that do not have at least one point in the cell boundary can be ignored and completely excluded from the pruned segments, that may reduce complexity in this function.


####################################################
Add a check in addSegmentWithDivisions so that newly created tangents for the start and end points are clamped within 60 degrees of the vector between the start and end point.

######################################################
*********************************************************
The latest build was working for some time but eventually produced an error:
12:32:34.181 [AWT-EventQueue-0] INFO dendryterra.DendrySampler -- Constellation: center=(-4.5, 25.5), size=3.0, ConstellationScale=1, cellRange=[-7,23 +5x5]
12:32:34.182 [AWT-EventQueue-0] INFO dendryterra.DendrySampler --   -> drafted=25, bounded=9, merged=9 stars (size=3.000, halfMerge=0.333, center=(-4.50,25.50), draftBounds=[-6.76,23.03 to -2.09,27.80])
ava.lang.NullPointerException: Cannot read field "x" because "v" is null
	at dendryterra.SegmentList.rotateVector(SegmentList.java:271)
	at dendryterra.SegmentList.computePointTangent(SegmentList.java:385)
	at dendryterra.SegmentList.computeTangentsForConnection(SegmentList.java:343)
	at dendryterra.SegmentList.addSegmentWithDivisions(SegmentList.java:236)
	at dendryterra.SegmentList.addSegmentWithDivisions(SegmentList.java:188)
	at dendryterra.DendrySampler.createSegmentV2(DendrySampler.java:2274)
	at dendryterra.DendrySampler.connectAndDefineSegmentsV2(DendrySampler.java:2060)
	at dendryterra.DendrySampler.CleanAndNetworkPointsV2(DendrySampler.java:1996)
	at dendryterra.DendrySampler.generateAsterism(DendrySampler.java:1300)
	at dendryterra.DendrySampler.generateAllSegments(DendrySampler.java:439)
	at dendryterra.DendrySampler.computeAllSegmentsForCell(DendrySampler.java:3630)
	at dendryterra.DendrySampler.evaluateWithPixelCache(DendrySampler.java:3607)
	at dendryterra.DendrySampler.getSample(DendrySampler.java:387)
	at com.dfsek.noise.swing.NoisePanel.getImage(NoisePanel.java:294)

Ref funcs:
connectAndDefineSegmentsV2

The intermediate points (knots) are not visible in the PIXEL_DEBUG return type for non-trunk level 0 segments or for the stitches.  Please investigate this issue.  I believe it could be an issue in the way the pixel cache is produced or an issue in point types not getting set / getting lost as segment lists are joined.

#########################################################

Some updates:

1. In pruneSegmentsToCell, make sure orphaned segments (segments where both end types are "EDGE") are subdivided in half so a single knot remains accessible for higher levels to attach to.

2. Level 2 + segments don't appear to be created, are level 1 segments getting made available for level 2 segments to attach to?

3. Somehow at level 1, jitter is significant enough to allow points to overlap, but this should not be the possible.  Is there a bug around "addSegmentWithDivisions" that allows the jitter to exceed 50% of the maxSegmentLength?

4. The overlap may be due to excessive tangents adopted from lower levels.  In "addSegmentWithDivisions" add a check to saturate / bound the magnitude of the tangent returned from computeTangentsForConnection to be no larger than the magnitude of maxSegmentLength.

##########################################################

Even with no jitter, level 1 segments are returning strange paths, this is due to how to the branches are getting built out.

Make sure distance for order to attached points is closest to original segment, but validation of distance is against fully defined segment.

##########################################################

Let's update connectAndDefineSegmentsV2 phase B.

Current implementation: continuously chaining in the shortest segment, each time re-evaluating and finding the next closest segment.

Updated implementation: Sort all unconnected points by their shortest distance to previous segment / trunk segment points.  Then loop from closest to furthest unconnected point, each time adding the point into the segment list, connecting it with the standard connection rules to available points in the segment list.  This way the initial sort does not get affected by newer points, but newer points are available to connect to each iteration.  This should reduce likelihood of new segments crossing.

###########################################################

Remove "SEGMENT_DEBUGGING" 40 and 50 in it's current form.  Now add debug 40 to perform the following:
 - return all level 1+ points as 0-length segments so they can be reviewed after merging.
 - do not perform the normal connection of those points back to the lower level segment, as the intent is to see how the level 1 points are distributed before they are joined to the previous level segment tree.

 Note currently there are multiple tangles due to close points on level 1+, I suspect there is a defect preventing level 1+ points from merging properly.


##########################################################

Update debug definition 40 so that it works on concurrent levels (level 1+)

Ex: If at level 1, just return the points as 0-length segments as done now.

If at level 2, fully connect level 1 (previous level) and then return level 2 points only as 0-length segments.

I want this change to help debug why I don't see any level 2 segments.

Also add logger context in debug 40 to indicate for the highest level (resolution) selected how many points were drafted, how many remained after merging, and how many remained after probabilistically removing.

##########################################################

Still not seeing any level 2 points come through debug mode 40 on level 2.  The tool indicates 5 points were retained after probabilistic removal.  Is there any other logic that is removing points before they are connected to previous level segments?

Unrelated:

Add logic to prevent hermite splines from crossing the cell boundary, this can be done by checking the tangent components, such that if the point x + or - the absolute tangent x component intersects the cell well, reduce it accordingly, and similarly for the y components.  Thus every created tangent component can have it's components "clamped" in a way to ensure they would not overlap the cell boundary.  This only applies to level 1+ tangent creation.  This can probably be done after boundTangentMagnitude in SegmentList.java.

In the cell pruning logic (pruneSegmentsToCell) prevent segment creation near edges by implementing the following:
- If after pruning, the segment chain is more than 4 segments long, set the two points closest to each edge to edge types to prevent higher level segments from connecting to them, and to reduce the likelihood new segments will cross the cell boundary.

#####################################################

I have commented markEdgePointsNearBoundary because it is converting more than 2 points.

Simplify the logic so that for each segment, if one point is already an edge type and that point only has one connection, and the connected segment to the non-edge point has two non-edge points, then update the non-edge point for the current segment to be an edge type.  This should prevent excessive conversion of points to edge types.

##########################################################


Refactor lines 454 to 458 of DendrySampler.java to generate points in a slightly more simplified manner.

Create a new function to create the points for the specified level at the specific query level.   (perhaps just refactor generatePointsForCellAtLevel(queryX, queryY, level))

Determine the number of rows or columns per cell as POINTS_PER_CELL[level]

Then similar to generatePointsForCellAtLevel, generate all the level points according to the points per cell grid spacing to get (points per cell)^2 number of points drafted for the cell.  But now the points should span the entire world cell from 0 to 1 or 2 to 3 ect according to the query cell, instead of a resolution that reduces the coverage of points.

##########################################################

It appears some level 0 points are not getting connected to the rest of the asterism (missing / dropped entirely).  This appears to be happening in phase B of connectAndDefineSegmentsV2, I expanded max level search to inf, and it's working, but this shouldn't have been an issue.

#################################################################

Increase the number of segments created for trunk (double the amount of segments created)

Increase the allowable tangent multiplier.... since these are tight connections.

Figure out why chained segments can't connect to each-other.

##########################################################

In phase B of CleanAndNetworkPointsV2 it appears 

##############################################

Add a new parameter "minimum" (DendryTemplate.java) with a default value of 0 that defines the elevation that level 0 points are forced to and add functionality so that that at level 0 the candidate points at level 0 are assumed to be at the "minimum" elevation to change drive changes to path preferences.

###########################################################

Update the following functions to all use a common function to calculate the normalized slope instead of duplicating the similar functionality.

findBestTrunkNeighborV2
findBestTrunkNeighborFromPoint
findBestNeighborV2

findBestTrunkNeighborV2 and findBestTrunkNeighborFromPoint have older slope calculations.

findBestNeighborV2 should have the latest updates for normalized slope calculation.

intermediateTangent

#####################################################


Unconnected points at level 0 do not appear to connect back to the trunk, or to each-other.  Is there a bug in level 0 asterism creation that is preventing new segments from connecting to points that were just added in previous segment iteration creation or at trunk creation?


Add new parameter inputs:
riverwidth - a sampler that gives a value that will be used to determine the river width at a given point.  Default 16.
borderwidth - a sampler that gives a value that will be used to determine the width of a border region around a river.   Default 20

Now add a new return type "PIXEL_RIVER" which will return the following using the pixel cache data:

1. For each query / coordinate of "PIXEL_RIVER" the river width and border width will be computed from their samplers as the noted parameter inputs above (the inputs above are in raw coordinates, so they need to be divided by the pixel cache value if distances will be compared using pixel units.)
2. For the resolution / levels that were configured, the riverwidth per level shall be calculated as:
RiverWidthAtLevel = riverwidth*(0.6^level), but will not be able to be less than 2x the pixelcache resolution.

Then, for each coordinate, an approximation will be used to determine if the coordinate is in range of a river knowing the pixel distance to each river type and the width of the river at a particlar river, and if not a river than determin if the query coordinate is a boundary using a similar method.

IMPORTANT: The accuracy of distance can be appoximated by using a "star" or set of lines from the query point to determine if it intersects a river pixel.  Alternatively, other matrix functions may be more suitable to perform operations across the entire cached pixel cell.

IMPORTANT: If a queried coordinate is within the riverwidth or borderwidth from the cell wall, the adjacent cell's pixel cache will need to be queried if no matches were found in the current cell.

A "within river border" response should always have priority over a border.

Returns 0 for river.
Returns 1 for border.
Returns 2 for not river.

#########################################################

In order to reduce memory consumed by SegmentList elements, rename the class "Segment3D" to "SegmentIdx" and remove the depricated "Point3D" fields in that class.

Unused / depricated funcitons need not be updated for compatability, they will be addressed on a case-by-case basis.

Then create a new class "Segment3D" in "Segment3D.java" which has the following fields:

    public final Point3D srt;  // Start point of the segment
    public final Point3D end;  // End point of the segment
    public final Vec2D tangentSrt;  // Tangent direction at start point (may be null)
    public final Vec2D tangentEnd;  // Tangent direction at end point (may be null)

This newer "Segment3D" class will be used in the further evaluations.

******************************************

Rename the current return type PIXEL_RIVER to PIXEL_RIVER_LEGACY, as I will describe an updated implimentation.

******************************************

Lets make some more significant changes which should speed up results for a new return "PIXEL_RIVER".

Add new sampler parameters to define how stored information will be normalized in this new method:
 - "max" defines the maximumm expected elevation / control function value to quantize results for.  Default 2.0, will be described further below.
 - "max-dist" defines the maximum expected distance computed, and affects how distance will be quantized for results.  Default 50.  If "max-dist" is less than default-riverwidth + defaultBorderwidth produce an error.

1. After evaluating a pruned cell for all levels (the returned value from generateAllSegments), store the full segmentlist for that pruned cell.  Retain 20 MB for retained segment lists, removing the oldest access segment lists if space needs to be freed.  Note now if information is needed from that cell, the entire call can be loaded instead of regenerating the full asterism and segment permutations.

2. Now, impliment a new cache for which each element is a fixed 256x256 size of blocks, where each block represents a pixel-cache sized square.  Each 256x256 grid here can be referred to as a bigchunk.
 - Each grid / bigchunk is located in world coordinates using 2 doubles or bigchunk coordinates using uint32.  If using bigchunk coordinates it's okay to produce an error when out of the 32-bit integer bounds.  (As the pixelcache value * 256 (the size of a bigchunk) is the real world coordinate where an error would be produced)
 - Each grid / bigchunk contains 2 values: a UInt8 normalized elevation, and a UInt8 normalized distance.
 - Normalized elevation should be initialized to 0.
 - Normalized distance should be initialized to 255.
 - Each grid / chunk may need to hold a flag or similar that indicates it has been computed.

Thus, each chunk should consume 256x256x2 = 131072 bytes + 8 byte locator + 4 byte age ~ 132 kb.

Similar to the existing cache, this cache should be allowed to consume up to 20 MB.
 
I expect similar functions as getOrCreatePixelCache and evaluateWithPixelCache, but this time to evaluate when a chunk when it is not available.

 3. Now a queried coordinate from the DendryTerra sampler for a PIXEL_RIVER return type can first determine if the pixel chunk exists in memory (if the bigchunk has been created that contains information for that queried coordinate).  If not, the bigchunk needs to be defined.

 To create / fully define a bigchunk:

    A. Load / get a pointer to all cells that might contain segment information needed to compute the bigchunk.  Any cell boundaries that are within the "max-dist" of the query point may contain segments that will influence the bigchunk.
    B. For each cell, convert the SegmentList elements into a list of Segment3Ds (as other context from the SegmentList class type is not necessary for the PIXEL_RIVER computations). Only segments that have a single point within "max-dist" for the bigchunk border need to be retained / kept as they are the only ones which will likely be able to influence the bigchunk.  Each Segment3D should also be accompanied by a uint8 level, which can also be taken from the SegmentList.
    C. Now all segments have been collected that could influence the bigchunk.  Loop through each segment:
        i. For the segment, compute the selected river width according to the nominal river width and the level.
        ii. Compute how many samples need to be taken along the hermite spline to effectively populate each box / pixel of the bigchunk. For each value t in the segment:
            a. If it's position is NOT within the boundary of the bigchunk (can just use manhatten distance), continue to the next evaluation along the segment..
            b. At each sample point on the spline (t), compute the next tangent, the current tangent, and the previous tangent (clamping the previous tangent to the current tangent on the first sample and the next tangent to the current tangent on the last sample.)
            c. The range / angle change in the 3 tangent vectors will define the "cone angle" where box / pixel information can be populated by the current segment.
            d. Now solve to populate all boxes that are influenced by the current segment partition from it's start point to it's end point.
            e. First, compute the potential elevation for the current point on the segment t by using linear interpolation bbetween the start point and the end point.  Then divide this elevation by the "max" parameter value, saturate it to 1, multiply it by 255, and store it as a the current segment elevation in uint8.
            f. Compute the location of the sample point on the bigchunk box and set it's distance to 0 and it's elevation to the segment elevation so long as it's potential elevation is greater than the current elevation of the box's current value.
            g. Compute the perpendiculars / cones of the current sample, moving outward along each perpendicular (of the current tangent) to see which box the sample should be allocated to over the "max-dist" in each direction.  At each step moving outward, the distance from the segment would increase linearly, and the cone arc can be calculated knowing the cone angle and the radius.  If the cone arc for the current perpendicular sample is greater than half of the the pixelcache value, multiple samples along the arc must also be taken.  It is important the cone is only considered for the side of the segment that is bowing from.  Following the rules to compute the normalized distance populate the box value in the bigchunk's cache.
                - If the distance / radius from the segment is less than the selected river width, the normalized distance will be calculated as the distance / selected river width.
                - Else the normalized distance will be calculated as the distance - selected river width.
                - If the normalized distance is less than current distance recorded in the corresponding box of the bigchunk, update the box with the lower distance of the current sample.
                - If the elevation is higher than the current recorded elevation in the corresponding box of the bigchunk, update the box with the lower distance of the current sample.

The returned value from "PIXEL_RIVER" can now return two values - the elevation and the distance, converted from their stored quantized uint8 values back to the standard return type.

The test cases should be updated so that PIXEL_RIVER_LEGACY and PIXEL_RIVER can be evaluated for comparison.


#########################################################

 A check should be made every time a point is added to a Segment list when above level 0, where if a node is getting attached and is lower elevation than the node it's connecting to, all nodes that it's connected to that chain down to the level 0 asterism should be reduced in elevation ratiometrically by the elevation of the point to add / elevation of the point that it's connecting to, so that river cannot flow "uphill".  This can likely be implimented within "SegmentList.java", where if a new point is lower than it's connecting point, a loop on the points and their connectios can be performed to ensure only points going down to 0 are reduced in height.

River return type - Input to specify the width for each level to report as river enum.  Input to point to sampler to define the width variation.  Refer to current river sampler, since "flats" or lowlands might be a special type that needs consideration?

############################################################3

1. Looking at sampleSegmentAlongSpline it appears to be performing linear interpolation for both point selection and tangent computation, but it should be using hermite interpolation for both so long as curvature is available and the splines option is set.  Please confirm.

2. The returned distance should be scaled according to the quantized resolution. The quanitize resolution is 255/max-dist.  This was not clarified before, so the distance stored in the cache should be the normalized distance * the quantize resolution and capped to uint8 max, and when the value is returned from the DendryTerra sampler, the uint8 value should be divided by the quantize resolution.

3. Since there is a single return type now, add another type "PIXEL_RIVER_CTRL" which returns the elevation.  Like distance, the elevation should also use it's quantize resolution to store in UInt8 and converted back when returned from the sampler.  So with a parameter max of 2.0 and min of 0.0, a UInt8 value of 0 should be stored for 0 elevation, and a UInt8 value of 255 should be stored for 2.0 elevation.

####################################################3

When I run PIXEL_DEBUG, I see multiple segment structures as expected verifying the segment tree is working as expected, but when I run PIXEL_RIVER, I only see a single dot in the coordinates 0,0, everywhere else is a color that represents 50 (the default max-dist parameter).  It seems there is a defect in PIXEL_RIVER, perhaps an issue in properly evaluating the correct coordinates or evaluating the segments, or in the normalized distance calculation which should be using defaultRiverwidth and defaultBorderwidth as real world distances to normalize against.  There should have been multiple 0-distance returns in PIXEL_RIVER, but there is only a single 0 distance getting returned.

#############################################################

Now I'm seeing the sampler return 0 or near 0 values continuously.  I think there are bigger semantic issues in the code causing confusion.

In the code there are multiple comments about a "world coordinate", but there are only 3 coordinate systems:

1. The sampler coordinate, the literal coordinate system is receiving queries from.  If anything this would be the world coordinate, but the code does not align with that.
2. The grid coordinates, this is the literal coordinates divided by the gridsize, and is used to normalize cellular information.  It appears sometimes these are referred to as world coordinates, incorrectly.
3. The bigChunk coordinates, which are located in the sampler coordinates at each 256*pixelcache increment.

Review the code and fix naming conventions to align with what is described above.

The parameter inputs for distance are defining distances in sampler coordinate system.  Review the code and ensure the parameters are properly converted for the distance they are being evaluated against, and rename relevant functions / comments to help clarify if they are acting on sampler coordinates, grid coordinates, or bigChunk coordinates.

###################################################

Note: When performing a test, to guarantee results (changing output values as a function of different sampled points), the gridsize must be set equal to or smaller than the span of the region being queried (so if the query is from x = 0 to x = 64, the gridsize should be 64 or smaller to guarantee segment generation at level 0 within the query region), and the pixelcache must be be no smaller than the resolution of the points being sampled. (If sampling at each 10 increments x=0, x = 10, x = 20, the pixelcache must be at least 10 or greater so information can be returned for the query region instead  of being hidden in an un-queried location (like x = 5 in this example)).  As the gridsize get's smaller, it will result in more level 0 segments being produced, which will likely significantly impact performance.

####################################################

Fixed?:

1. Returned distance (PIXEL_RIVER) does not appear to be correctly converted back to sampler / world coordinates.  I would expect to see the max distance (max-dist) returned for some samples after converting back from the quantized UInt8 distance in the BigChunk.  With a gridspacing of 2000 I see a max value of 0.025 returned now with PIXEL_RIVER, but the max value returned should be 50.  It appears the gridsize is not getting multiplied by the value to return the value  in sampler units.

2. Space within the river is not 0 to 1, are we not normalizing the inner part of the river correctly?  The part within the river boundaries?

########################################################

Goals:
 - Fill in missing sections from the ends of segments.
 - Fill in pin-holes in the returned values where some boxes aren't getting values populated, likely due to the way positions are rounded around arcs.
 - Smooth out elevation changes to have more of a curved shape flowing toward the downward slope of a river / segment flow, with "straighter" falloff as the slope increases.
 - Make sure transitions to new segments are relatively clean.

Make additional updates to the RIVER_PIXEL solver:

1. Initialize BigChunk box elevation or control values to UInt8 max instead of 0.

2. When collecting segments to evaluate for PIXEL_RIVER, also add 2 sister lists that contains the number of connections for each segment's end connection and start connection.

3. When iterating over segments for PIXEL_RIVER, go from highest level segments down.  (Sort segment  before evaluating).  Note any sister arrays (level, start connections, end connections) must be evaluated in the same order per corresponding segment.

4. Update functions that set the quantized elevation to directly use the quantized elevation value as an input instead of scaling as a part of it's function, as the quantized elevation must be known for other logic when evaluating a segment.

5. Make various updates to the ways segments are sampled and boxes for BigChunks are solved.  Calculate the slope of the segment as it's euclidean distance / change in height and for each evaluation of t (for loop below "// Sample along the segment"):

    A. If this is the first segment position (i=0), or if the previous sampled segment points was out of bounds:
     - initialize 3 UInt8 elevation variables for the outer, inner, and central river elevation to "elevation * elevQuantizeRes".  T
     - initialize 3 float elevation radius variables for the outer, inner, and central river elevation to 0.
     - compute the previous tangent and current tangent as the current tangent of the segment.
     - compute the previous and current position as the current position.
     - consider this the start of a new segment stream (boolean)

    B. Else this sample is not a part of a new segment stream (it is a continued segment stream from the previous iteration)
     - The previous tangent and previous position should be retained from the last solved sampler sample to compare to the current sample, that is previous tangent = tangent, previous position = position

    C. Perform boundary checks as is today to continue to the next sample if the current sample is out of bounds, else compute the other properties of the sample:
     - The point will be calculated as normal for the new sample.
     - A potential elevation will be computed as "elevation * elevQuantizeRes", where elevation is calculated the same way it is today, interpolated along the hermite spline.
     - If the potential elevation is different from the central river elevation, this means the quantized elevation has changed and all elevations must be updated:
        - The outer elevation = inner elevation
        - Inner elevation = central elevation
        - Central elevation = potential elevation.
        - Outer elevation radius = inner elevation radius.
        - Inner elevation radius = central elevation radius.
        - Central elevation radius = 1.
        - Set a flag indicating the elevation changed this sample.

    D. If the sample is a part of a new segment stream or if the tangent vs the previous tangent difference exceeds an absolute difference of 90 degrees or the distance between the current sample position and the previous sample position exceeds 70% of the pixelcache /gridsize distance or this is the last sample in the segment or if the elevation changed this sample, the sample should be used to populate the bigChunk boxes similar to how it is done today, else the loop should continue to evaluate the next sample.

    E. Update the elevation radii :
        - if the elevation did NOT change this sample all elevation radii will be decremented by the distance to the previous evaluated point, with a minimum value of 0.
        - For all elevation radii, saturate them to the (distance to the end segment point - 1), this will ensure the elevation radii is forced to 0 as the segment reaches 0, which will prevent elevation artifacts from overlapping lower level rivers inconsistently.

    F. If the segment sample is at the start or end and the number of connections is 1, set a fill flag that will be used by projectConeToBoxes.

    F. Update projectConeToBoxes:
        i. Will now only require current tangent and previous tangent, in addition to all the new elevation parameters and the fill flag.
        ii. calculateConeAngle and calculateBowDirection also only require current and previous tangent.
        iii. If the fill flag is set, compute the cone angle as 180 degrees clockwise, set the perpendicular as 90 degrees clockwise from the current tangent for start points and 90 degrees counterclockwise for end points.  This way the entire semicircle surrounding the end of the point will be filled.
        iiii. For each step outward, if there is an elevation radius that is non-zero and the normalized distance is less than 1, determine which elevation should be applied.  For each non-zero elevation radii from biggest to smallest:
            - Calculate the samples distance to the elevation centroid (centroid distance) as sqrt(normalized outward distance ^2 + (selected elevation radius * min(0,(1- current slope / max slope))) ^2).
            If this value is less than one, the selected elevation will be the elevation corresponding to the elevation radius evaluated.
            - Else continue to the next largest elevation radius.
            - If no centroid distances are less than 1, ues the central elevation.
        v. For each step outward, if the step is not the first step outward, set the flag that will allow adjacent box filling for the updateBox function.  This should be implemented with a compile-time option that allows the flag to be disabled so it can be evaluated with / without adjacent box fills. 
5. Update the updateBox function so that it has a flag input indicating if the 4 adjacent boxes should be filled with the same value (excluding any adjacent boxes that are outside the edge of the bigchunk).  This flag may be evaluated la

5. If elevation for box is currently greater than what it is about to be updated to and it was previously set (check distance < 255), and the box is getting set for level 1+ or level 0+ and the outward step is 1 pixelcache distance away from the river edge add some random flat noise that will allow the elevation set in the box to be any value between it's current value and the value that would have been set.

###############################################################################################

Reminder: Final ToDo:

Remove all unused / legacy functions.
Remove unnecessary fields in Point3D:
    public final PointType srtType;
    public final PointType endType;
    public final Point3D srt;
    public final Point3D end;
