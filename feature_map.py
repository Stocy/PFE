face_pts_dict = {}
edge_pts_dict = {}
vertex_pts_dict = {}

for i in range(n_pts):
	type = [dst[2][0][3] for dst in dsts][i]
	index = [dst[2][0][4] for dst in dsts][i]
	if (type != 'Face'):
		faces = []
		if (type == 'Edge'):
			edge = cube.Edges[index]
			if edge in edge_pts_dict:
				edge_pts_dict[edge].append(i)
			else : edge_pts_dict[edge] = [i]
			faces = cube.ancestorsOfType(edge,Part.Face)
		if (type == 'Vertex'):
			vertex = cube.Vertexes[index]
			if vertex in vertex_pts_dict:
				vertex_pts_dict[vertex].append(i)
			else : vertex_pts_dict[vertex] = [i]
			faces = cube.ancestorsOfType(vertex,Part.Face)
	else :
		face = cube.Faces[index]
		if face in face_pts_dict:
			face_pts_dict[vertex].append(i)
		else : face_pts_dict[face] = [i]
	print(len(faces))