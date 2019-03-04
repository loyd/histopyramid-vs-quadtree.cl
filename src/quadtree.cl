typedef float3 point_t;

typedef struct {
    float4 bbox;
    atomic_int used;
} shared_t;

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

uint alloc_node(nodeptr_t restrict quadtree, shared_t *restrict shared, int lock) {
    uint idx = atomic_fetch_add(&shared->used, 1);

    nodeptr_t node = quadtree + idx;
    atomic_store(&node->lock, lock);
    node->type = EMPTY;
    node->count = 0;
    node->value = (point_t)(0.);
    node->quarters[0] = node->quarters[1] = node->quarters[2] = node->quarters[3] = 0;

    return idx;
}

bool initial_stage(shared_t *shared) {
    return atomic_load(&shared->used) == 0;
}

void lock(nodeptr_t node) {
    while (atomic_exchange(&node->lock, LOCKED) == LOCKED) {}
}

void unlock(nodeptr_t node) {
    atomic_store(&node->lock, UNLOCKED);
}

void insert(nodeptr_t restrict quadtree, point_t point, shared_t *restrict shared) {
    float4 bbox = shared->bbox;
    float2 shape = bbox.zw - bbox.xy;
    float2 origin = .5f * (bbox.xy + shape);

    point_t extra;
    bool has_extra = false;
    nodeptr_t node = quadtree;

    lock(node);

    if (initial_stage(shared)) {
        alloc_node(quadtree, shared, LOCKED);
    }

    for (uint level = 0; level < MAX_DEPTH - 1; ++level) {
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
            node->quarters[qno] = alloc_node(quadtree, shared, UNLOCKED);
        }

        if (has_extra) {
            int2 cmp = -isgreater(extra.xy, origin);
            uint extra_qno = cmp.x << 1 | cmp.y;

            if (extra_qno != qno) {
                uint quarter = node->quarters[extra_qno] = alloc_node(quadtree, shared, UNLOCKED);
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
    global const point_t *restrict points,
    global node_t *restrict quadtree,
    global shared_t *restrict shared
) {
    int gid = get_global_id(0);
    insert(quadtree, points[gid], shared);
}
