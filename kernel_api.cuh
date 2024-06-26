struct 
{
    dim3 gridSize = 1024;
    dim3 blockSize = 128;
} params;  


 void ComputeBoundingBox(int *mutex, float *x, float *y, float *z, float *left, float *right, float *bottom, float *top, float* front, float *back, int p_count);
 
 void ConstructOctree(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index, float *left, float *right, float *bottom, float *top, float* front, float *back, int p_count);
 
 void ComputeBodyInfo(float *x, float *y, float *z, float *mass, int *index, int p_count);
 
 void SortBodies(int *count, int *start, int *sorted, int *child, int *index, int p_count);
 
 void CalculateForce(float* x, float *y, float * z, float *vx, float *vy, float *vz, float *ax, float *ay, float *az, float *mass, int *sorted, int *child, float *left, float *right, int p_count, float gravity);
 
 void UpdateParticles(float *x, float *y, float *z, float *vx, float *vy, float *vz,  float *ax, float *ay, float *az, int n, float dt, float damp);
 
 void PopulateCoordinates(float *x, float *y, float *z, float *out, int p_count);
 
 void ResetArrays(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start, int *sorted, int *child, int *index, float *left, float *right, float *bottom, float *top, float* front, float *back,int p_count, int node_count);

