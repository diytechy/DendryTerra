CleanAndNetworkPoints:
Inputs:
    - New point cloud - 3d points generated from the cell grid that need to be linked into the main segment list (The 3d points, the gridspacing of the cells used to generate the 3d points or similar context giving parameter)
    - The segment list structure (if applicable, null or similar if called for asterisms before the trunk is created)
Outputs:
    Updated segment definitions
        The updated segment structure list with the relevant points and segments added.

Function setup:
    A. Determine the cell specific merge distance by multiplying "merge point spacing" with the gridspacing of the cells used to generate the points.
    B. Determine the maximum segment distance in a similar manner based on gridspacing.

* Clean any points: If network points is being performed for level 1 or higher, any points in the new point cloud should be merged according to mergePointsByDistance.
    Note this won't affect stars.
* Then remove all input points that are within the merge point distance of any other segments 1 level below (linear interpolation for segment distance or other alternatives are appropriate)
    Note this won't affect stars, since they are at the lowest level.
* Finally, if not at level 0, loop through each point and deterministically remove a point depending on both:
    1. It's percentage chance of removal according to it's probability of removal (1 - branchesSampler(x,z))
    2. It's percentage chance of removal according to it's distance from the input list segment structure.

* Now connect all the points (connectAndDefineSegments).

    IMPORTANT: Fetching points to find connections for in CleanAndNetworkPoints should only include points from the input new point cloud.  Neighbor points / candidates to connect to only include points from segments already inside the segment list structure except for the first segment creation (only occurs once for a single point).  After any segment creation operation, the point added into the segment must be removed or marked as removed from the input point cloud and added to the segment list structure.  After any subdivision operation, the divided segment must be removed and the new segments and their new points must be added to the segment list structure.

    A. If at level 0, build the "trunk" through the point cloud:
    While the trunk is incomplete:
        Starting at the highest elevation point:
            Attempt to create a connection using the "Connection rules" below to create a segment from the point noting the connection rules should be for a trunk, the segment should be fully defined using the connection rules below before continuing on.  The first iteration here will be special in that it will allow the first segment to be created.
            If no connections can be made, consider the trunk complete.
            Else if a connection is made, continue to attempt to extend the trunk from the previous neighbor that was connected.  If the point is added, it should be removed from the input point cloud as the point should now exist inside the segment list structure.
    IMPORTANT: Add a debug here to exit connections of the points early so that the trunk segment is returned along with 0-length segments for all unconnected points so the initial tree creation can be viewed. (SEGMENT_DEBUGGING==15)

    B. Now grow the input segment list structure until there are no points inside the input point cloud:
        i. Get all points from the input point cloud that are within the maximum segment distance of the points from the input segment list (ignoring any points that are an EDGE type).
        ii. Select the point that is closest to a segment point, and attempt a connection using "Connection rules" below.  Again, if the point is added, it should be removed from the input point cloud as the point should now exist inside the segment list structure.

    Now all points have a return path or have been removed if they did not have a return path.

    If this was performed for level 0 (asterisms), the elevation of all segment points should be shifted down to 0.

    Any points that are only assigned to a single segment should be marked / set to a "LEAF" type.

    #########################################################

Connection rules (for creating a connection and detailing the segment):

    GOALS: 
        - Connect all nodes within the network.
        - Make smooth transitions / paths.
        - When branching, branch with a more outward angle.
        - Prefer flowing "downward".
        - Give "twist" to flow, increase twist at flatter slopes.
        - For non-star segments, increase the likelihood that segments will be dropped as they approach the cell borders far from the root path, just to reduce the obvious square shaped boundaries that will naturally form otherwise.

    NOTE: It is expected all segments will have a flow direction from the start to the end point, where the start tangent is the angle the segment projects from, and the end tangent is the angle the segment flows into at the end point.

    1. Get all the neighboring non-edge points from the segment list structure that are in range (overhead xz distance less than the maximum segment distance) and get their properties:
        Calculate the normalized slope as ((height distance between the current point and the neighbor) / (overhead xz distance between current point and neighbor)^(DistanceFalloffPower)).  Here the DistanceFalloffPower can be 2, and helps to flatten out distant neighbors slopes to prefer tighter connections.
    2. If no neighbor is valid, return empty or otherwise communicate to the caller that no valid neighbor is found:
        If this is for tree creation, a valid neighbor is only where a normalized slope is negative.
        If this is for level 1 or higher, a valid neighbor is only where a normalized slope is less than lowestSlopeCutoff.
        Else a valid neighbor must exist, if we get to this point, log it since this condition should not be possible.
    3. Select the neighbor based on the following priority:
        A. The neighbor with the smallest true distance for any whose true distance is less than the merge distance.
        B. The lowest normalized slope.
        NOTE: If no neighbor is selected, exit this indicating no new connections were made because all neighbors are interconnected.
    4. If the selected neighbor is already connected with 3 nodes (point connection property), or the resulting segment would cross an existing segment, subdivide the closest segment from the segment list structure, and select that new knot from subdivision as the selected neighbor, continuing with the checks below.  Note the new point created by subdividing a segment should inherit the level of the segment that was subdivided and the tangent and the subdivision location, and the point type should be a knot. (This follows the standard subdivision rules / functions that already exist, and it should be reused here if applicable)
        NOTE: This should be incredibly rare due to subdivisions / displacement already present on the segment.  Add a log entry if this happens.
    5. Else if the selected neighbor already has a line passing through (the neighbor is already connected to 2 or more other points), set a property to indicate it needs to be merged as a branch which will be passed when creating the segment described below.

    If a connection is made when a slope is positive (greater than 0) the greater point and all it's downstream connections from other segments (including all lower levels) must be ratiometrically reduced in elevation so that "upward" flow does not occur, but the ratiometric change cannot reduce below 0.

    Now a neighbor has been selected, it now should have a start point (srt) defined as the original point, and the end point (end) as the selected neighbor.

    Now determine the tangents for the new segments:
        For each side of the segment that is not defined:
            If the point is not a branch into another node (set above, should only apply to end points for the segment being created):
                If the point is already connected to another segment, match the tangent of the connected segment for continuity.
                    If the points flow through (a start point connects to an end point or an end point connects to a start point) their connecting point tangents should be identical.
                    Else if two end-points are connecting or two start-points are connecting, the undefined tangent should be set to the inverse of the their connecting tangent to guarantee continuity.
                Else if the point is not connected to any other segment:
                    Determine a twist to apply to the point:
                        random deterministic value between +/-50 deg * max((1-abs(pointslope)/SlopeWhenStraight ),0)
                Determine the nominal tangent:
                    The point slope tangent will be calculated as the average between the angle from the start point to the end points and the slope tangent (note: this should be the cross product)
                Set the tangent of the point to be the nominal tangent rotated by the twist angle.
            Else for remaining segments that are entering as branches:
                The branch tangent will be a random offset of 110 to 170 degrees from the tangent of the other segments that are already connected to this point, on the same side as the other point of the segment.

    For each new tangent, verify the vector is within 60 degrees of the segment tangent (start to end angle).  If it is not, saturate the new tangent to those bounds.
    
    Finally, the segment should be subdivided according to the subdivisions per level (can be hard-coded per level) and displaced as should be done in subdivideAndAddPoints .  The new segments should be added into the segment pool list so they can be used / connected to as a part of CleanAndNetworkPoints.

Merge Rules:
- Affects "mergePointsByDistance"
IMPORTANT: This may be disabled for stars to ensure star availability in each cell.

    Perform the following until there are no distances less than the merge point distance:
    1. For all points, get the distance to any other point.
    2. Select the point that has the most connections / distances to other points that are below the merge point distance.  This selected point will be the epicenter for the current merge iteration.
    3. Select both the epicenter point and any of it's points whose distance are less than the merge distance.
    4. Remove all those points, and replace them with a point that is located at the average position.  Perform the control function to get the new elevation.
    5. Now rerun the check as necessary until there are no distances between points that are less than the merge distance.