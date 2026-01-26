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
Files to Create
1. build.gradle.kts

plugins {
    java
}

group = "com.dfsek.terra.addons"
version = "1.0.0"

repositories {
    mavenCentral()
    maven { url = uri("https://maven.solo-studios.ca/releases") }
}

dependencies {
    compileOnly("com.dfsek.terra:manifest-addon-loader:1.0.0-BETA+a159debe3")
    compileOnly("com.dfsek:seismic:0.8.2")
    compileOnly("com.dfsek.terra:base:7.0.0-BETA+a159debe3")
    compileOnly("com.dfsek.tectonic:common:4.2.1")
    compileOnly("org.slf4j:slf4j-api:2.0.9")
}

java {
    toolchain { languageVersion.set(JavaLanguageVersion.of(21)) }
}

tasks.jar {
    archiveBaseName.set("DendryTerra")
}
2. settings.gradle.kts

rootProject.name = "DendryTerra"
3. terra.addon.yml

schema-version: 1
contributors:
  - DendryTerra Contributors
id: dendry-noise
version: 1.0.0
entrypoints:
  - "com.dfsek.terra.addons.dendry.DendryAddon"
license: GNU General Public License v3.0
4. DendryAddon.java
Implements AddonInitializer
Uses @Inject for Platform and BaseAddon
Registers DENDRY sampler in ConfigPackPreLoadEvent
Registers DendryReturnType loader
Pattern: NoiseAddon.java
5. DendryTemplate.java
Extends appropriate base template or implements ObjectTemplate<Sampler>
Configuration fields with @Value and @Default:
n (int, default 2): Resolution levels 1-5
epsilon (double, default 0): Point bias [0, 0.5)
delta (double, default 0.05): Displacement amount
slope (double, default 0.5): Valley/cliff factor
frequency (double, default 0.001): Grid scaling
return (DendryReturnType, default ELEVATION)
sampler (Sampler, optional): Control function
salt (long, default 0): Seed modifier
Pattern: PseudoErosionTemplate.java
6. DendrySampler.java (Core Algorithm)
Implements com.dfsek.seismic.type.sampler.Sampler with methods:

getSample(double x, double z) - 2D sampling
getSample(double x, double y, double z) - 3D sampling (use x,z plane)
Algorithm Flow (ported from noise.h):

Scale input coordinates by frequency
Determine grid cell at resolution level 1
Generate neighboring points (3x3 grid) with epsilon bias
Evaluate control function for elevation at each point
Create segments connecting each point to lowest neighbor
Subdivide segments using Catmull-Rom splines (4 subdivisions)
Displace segment midpoints by delta amount perpendicular to direction
Repeat for resolution levels 2 through n (cell size halves each level)
Compute result based on return type:
DISTANCE: Euclidean distance to nearest segment
WEIGHTED: Distance weighted by branch level
ELEVATION: Interpolated Z from nearest segment
Key Methods to Port:

GeneratePoint() - Deterministic point in cell using epsilon
GenerateSegments() - Connect to lowest neighbor
SubdivideSegments() - Catmull-Rom smoothing
DisplaceSegments() - Perpendicular perturbation
NearestSegmentProjectionZ() - Find closest segment
7. Math Classes
Port from math2d.h, math3d.h, spline.cpp:

Point2D/Point3D: Immutable with add, subtract, scale, distance
Vec2D: Direction vector with dot, normalize, rotateCCW90
Segment2D/Segment3D: Line segments with midpoint, length
MathUtils: lerp, clamp, pointLineProjection, distanceToSegment
CatmullRomSpline: Chordal parameterization for smooth subdivision
Implementation Order
Setup Gradle project - build.gradle.kts, settings.gradle.kts, wrapper
Create math utilities - Point2D, Point3D, Vec2D, Segment classes, MathUtils
Implement CatmullRomSpline - Subdivision algorithm
Create DendryReturnType enum
Implement DendrySampler - Core algorithm (start with resolution=1)
Create DendryTemplate - Configuration binding
Create DendryAddon - Registration and initialization
Create terra.addon.yml - Manifest
Build and test - Place JAR in NoiseTool addons folder
Verification
Run gradlew.bat build to compile
Copy build/libs/DendryTerra-1.0.0.jar to C:\Projects\NoiseTool\build\libs\addons\
Start NoiseTool and verify console shows addon loaded
Test with YAML config:

sampler:
  type: DENDRY
  n: 2
  epsilon: 0
  delta: 0.05
  frequency: 0.001
  return: elevation
Visualize output to confirm branching patterns
Key Reference Files
File	Purpose
NoiseAddon.java	Addon registration pattern
PseudoErosionTemplate.java	Template with sampler field
noise.h	Core C++ algorithm
spline.cpp	Catmull-Rom implementation
math2d.cpp	Distance calculations

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