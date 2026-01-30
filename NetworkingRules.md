CleanAndNetworkPoints:
Inputs:
    - Unique cell location definition (The 3d points, the gridspacing of the cells used to generate the 3d points or similar context giving parameter)
    - Previous / lower level segments (if applicable, null or similar if called for asterisms, are the lowest level segments)
Outputs:
    Updated segment definitions
        Each segment has two 3d points (x,y,z), and two tangents describing the end condition in the x,z coordinates.

Function setup:
    A. Determine the cell specific merge distance by multiplying "merge point spacing" with the gridspacing of the cells used to generate the points.
    B. Determine the maximum segment distance in a similar manner based on gridspacing.

* Clean any network points: if any points are within the merge distance, merge the points to their average x,z position and resample the control function to determine their y height.
    Note this won't affect stars, since merging already happened.
* Then remove all input points that are within the merge point distance of any other segments 1 level below (linear interpolation for segment distance or other alternatives are appropriate)
    Note this won't affect stars, since they are at the lowest level.
* Now the number of points that must be connected can be summed, and referenced below to determine when all points have a path out of an asterism.  This is the number of points to connect this level.
* Finally, if not at level 0, loop through each point and deterministically remove a point depending on it's percentage chance of removal according to it's probability of removal (1 - branchesSampler(x,z))
* Now connect all the points:
    A. Loop through each point from highest elevation to next to the lowest elevation point to create initial downstream flows: 
        Attempt to create a connection using the "Connection rules" below to create a segment from the point, the segment should be fully defined using the connection rules below before iterating onto the next point to prevent overlapping connections.  For this reason this may not be able to be implemented as a simple for loop, but perhaps as a while loop that runs while the number of points without any connection is greater than 0, each time running the connection rules on the highest elevation point without any connection. 
            IMPORTANT: If greater than level 0 / asterism, and the lowest slope is greater than LowestSlopeCutoff, the selected point and any point connected to it shall be removed, since it cannot achieve a path back to the main asterism segment.
    B. Now that initial downflows have been created, there should be multiple "chains" that contain various numbers of interconnected segments.  While any set of points or segments remain unconnected to the same or lower level segment (so stars must all eventually have a path to each-other, since there is no level below them), find that chains escape path to the root path / asterism segments:
        ii. Select the chain with the fewest number of segments (excluding the segment chain that includes segments created at lower levels).
            Iterate through all nodes on the chain from the lowest to highest attempting to create a connection using the "connection rules" below.  Again, the segment should be fully defined using the connection rules below before iterating onto the next point to prevent overlapping connections.
            If a connection is made to another chain, exit the loop and reassess.  Else if the chain is never able to be connected back to another chain segment, remove the chain with it's segments and points since it has no return path.

    Now all points have a return path or have been removed if they did not have a return path.

    If this was performed for level 0 (asterisms), the elevation of all segments should be shifted down to 0.

    #########################################################

    

Connection rules (for creating a connection and detailing the segment):
    i. Find the neighboring point (overhead xz distance less than the maximum segment distance) with the lowest slope (difference in y height / overhead distance in xz coordinates) that is also NOT connected to the current point through another path.  If no potential connection is found, exit this indicating no new connections were made.
    ii. If neighbor is already connected with 3 nodes, or the resulting segment would cross an existing segment, subdivide the closest spline (at the same or 1 level down), and pin the new node, continuing with the checks below.  Note the new point created by subdividing a spline should inherit the level of the spline that was subdivided and the tangent and the subdivision location.
        NOTE: This should be incredibly rare due to subdivisions / displacement already present on the segment.  Would debugs be useful here?
    iii. Else if neighbor already has a line passing through (the neighbor is already connected to 2 or more other points), set a property in the segment to indicate it needs to be merged as a branch after that line's tangent has been computed.
    iv. Else if the neighbor already is connected to one point and the neighbor already has a defined tangent (which would only be the case for lower level points), connect to neighbor matching the tangent for a continuous flow.
    v. Else if the neighbor already is connected to one point, connect to the neighbor, and set the tangent of the neighbor to be the direction between it's two connected points.

    If a connection is made when a slope is positive (greater than 0) the greater point and all it's downstream connections should be ratiometrically reduced in elevation so that "upward" flow does not occur.

    Else if the segment exists, it now should have a start point (srt) defined as the original point, and the end point (end) as the lowest slope neighbor.

    Now determine the tangents for the new segments:
        For each side of the segment that is not defined:
            If the end is NOT a branch into another node (set above):
                Determine a twist to apply to the point:
                    random deterministic value between +/-70 deg * max((1-slope/SlopeWhenStraight ),0)
                If the point is connected to two other points that aren't coming from segments that are set to branch into the selected point, the nominal tangent is the tangent between those two points.
                If the point is only connected once, the nominal tangent is the tangent of the point itself.
                Set the tangent of the point to be the nominal tangent rotated by the twist angle.
                If the end of a segment is set to branch into another path, wait to solve it until after all other segment tangents are defined.
            Else for remaining segments with branch ends:
                The branch tangent will be a random offset of 110 to 170 degrees from the flow path tangent on the same side as the other point of the segment that has already been defined for the same point on two other segments creating the main path.

    Finally, the segment should be subdivided according to the subdivisions per level (can be hard-coded per level), displaced, and assigned small adjustments in their tangents to give more distinct curvatures and to make available points to future calls.  The subdivision points should be added into the point list so they can be used / connected to as a part of CleanAndNetworkPoints