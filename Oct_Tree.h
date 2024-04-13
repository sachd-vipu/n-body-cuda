
struct Position{
    float x;
    float y;
    float z;
};

struct Velocity{
    float vx;
    float vy;
    float vz;
};

struct Body{
    Position position;
    Velocity velocity;
    float mass;
};

struct Node{
    float center_of_mass;
    float accumulatedMass;
    float minCorner;
    float maxCorner;
    Node* children[8];
    Body* body;
};

struct Oct_Tree
{
    Node* root;
    int maxDepth;
    int maxBodies;
    float theta;
};
