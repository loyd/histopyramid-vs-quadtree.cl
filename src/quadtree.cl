typedef float3 point_t;

typedef enum {
    EMPTY = 0,
    LEAF,
    BOX,
} node_type_t;

#define UNLOCKED 0
#define LOCKED 1

typedef struct {
    atomic_int lock;        // {UNLOCKED, LOCKED}
    node_type_t type;

    uint count;
    point_t value;

    /*
     *  ^
     *  |
     *  +-------+-------+
     *  |       |       |
     *  |   1   |   3   |
     *  |       |       |
     *  +-------+-------+
     *  |       |       |
     *  |   0   |   2   |
     *  |       |       |
     *  +-------+-------+---->
     */
    uint quarters[4];
} node_t;

typedef volatile node_t *nodeptr_t;

uint alloc_node(nodeptr_t restrict quadtree, atomic_int *restrict next_free, int lock) {
    uint idx = atomic_fetch_add(next_free, 1);

    nodeptr_t node = quadtree + idx;
    atomic_store(&node->lock, lock);
    node->type = EMPTY;
    node->count = 0;
    node->value = (point_t)(0.);
    node->quarters[0] = node->quarters[1] = node->quarters[2] = node->quarters[3] = 0;

    return idx;
}

inline bool initial_stage(atomic_int *next_free) {
    return atomic_load(next_free) == 0;
}

inline void lock(nodeptr_t node) {
    while (atomic_exchange(&node->lock, LOCKED) == LOCKED) {}
}

inline void unlock(nodeptr_t node) {
    atomic_store(&node->lock, UNLOCKED);
}

void insert(
    float4 bbox,
    uint max_depth,
    nodeptr_t restrict quadtree,
    point_t point,
    atomic_int *restrict next_free
) {
    float2 shape = bbox.zw - bbox.xy;
    float2 origin = .5f * (bbox.xy + shape);

    point_t extra;
    bool has_extra = false;
    nodeptr_t node = quadtree;

    lock(node);

    if (initial_stage(next_free)) {
        alloc_node(quadtree, next_free, LOCKED);
    }

    for (uint level = 0; level < max_depth - 1; ++level) {
        if (node->type == LEAF) {
            extra = node->value;
            has_extra = true;
            node->type = BOX;
        }

        ++node->count;
        node->value += point;

        if (node->type != BOX && has_extra) {
            ++node->count;
            node->value += extra;
            node->type = BOX;
        }

        if (node->type == EMPTY) {
            node->type = LEAF;

            unlock(node);
            return;
        }

        int2 cmp = -isgreater(point.xy, origin);
        uint qno = cmp.x << 1 | cmp.y;

        if (!node->quarters[qno]) {
            node->quarters[qno] = alloc_node(quadtree, next_free, UNLOCKED);
        }

        if (has_extra) {
            int2 cmp = -isgreater(extra.xy, origin);
            uint extra_qno = cmp.x << 1 | cmp.y;

            if (extra_qno != qno) {
                uint quarter = node->quarters[extra_qno] = alloc_node(quadtree, next_free, UNLOCKED);
                nodeptr_t inner = quadtree + quarter;
                inner->type = LEAF;
                inner->count = 1;
                inner->value = extra;

                has_extra = false;
            }
        }

        nodeptr_t inner = quadtree + node->quarters[qno];
        lock(inner);
        unlock(node);
        node = inner;

        int2 dir = 2 * cmp - 1;

        shape *= .5f;
        origin += .5f * convert_float2(dir) * shape;
    }

    ++node->count;
    node->value += point;
    node->type = LEAF;

    if (has_extra) {
        ++node->count;
        node->value += extra;
    }

    unlock(node);
}

kernel void run(
    float4 bbox,
    uint max_depth,
    global const point_t *restrict points,
    global node_t *restrict quadtree,
    global atomic_int *restrict next_free
) {
    int gid = get_global_id(0);
    insert(bbox, max_depth, quadtree, points[gid], next_free);
}
