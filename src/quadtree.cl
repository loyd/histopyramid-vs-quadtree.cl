#define info(msg) printf("%d: " msg "\n", get_global_id(0))
#define info1(msg, a) printf("%d: " msg "\n", get_global_id(0), (a))
#define info2(msg, a, b) printf("%d: " msg "\n", get_global_id(0), (a), (b))
#define info3(msg, a, b, c) printf("%d: " msg "\n", get_global_id(0), (a), (b), (c))
#define info4(msg, a, b, c, d) printf("%d: " msg "\n", get_global_id(0), (a), (b), (c), (d))
#define assert(x) do { if (!(x)) info1("ASSERT %s", #x); } while (false)

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

uint alloc_node(nodeptr_t restrict quadtree, shared_t *restrict shared) {
    uint idx = atomic_fetch_add(&shared->used, 1);

    nodeptr_t node = quadtree + idx;
    node->type = EMPTY;
    node->count = 0;
    node->value = (point_t)(0.);
    node->quarters[0] = node->quarters[1] = node->quarters[2] = node->quarters[3] = 0;

    info1("alloc %d", idx);

    return idx;
}

bool initial_stage(shared_t *shared) {
    return atomic_load(&shared->used) == 0;
}

static nodeptr_t first = NULL;
#define rel(n) (n - first)

void lock(nodeptr_t node) {
    if (first == NULL) {
        first = node;
    }

    info1("locking %d", rel(node));

    int n = 100000;

    while (atomic_exchange(&node->lock, LOCKED) == LOCKED) {
        if (--n == 0) {
            info1("LOCK FAILED %d", rel(node));
            return;
        }
    }

    info1("locked %d", rel(node));
}

void unlock(nodeptr_t node) {
    atomic_store(&node->lock, UNLOCKED);
    info1("unlocked %d", rel(node));
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
        alloc_node(quadtree, shared);
    }

    for (uint level = 0; level < MAX_DEPTH; ++level) {
        info4(">> %.2v2f %.2v2f %d %d", origin, shape, rel(node), node->type);

        if (node->type == LEAF && level < MAX_DEPTH - 1) {
            extra = node->value;
            has_extra = true;
        }

        ++node->count;
        node->value += point;

        if (has_extra) {
            node->type = BOX;
        } else if (node->type != BOX) {
            assert(node->type == EMPTY || level == MAX_DEPTH - 1);

            if (level == MAX_DEPTH - 1) {
                info("TOO DEEPLY");
            }

            info3("placed %.2v2f %.2v2f %.2v2f", point, origin, shape);
            node->type = LEAF;

            unlock(node);
            return;
        }

        int2 cmp = -isgreater(point.xy, origin);
        uint qno = cmp.x << 1 | cmp.y;

        if (!node->quarters[qno]) {
            node->quarters[qno] = alloc_node(quadtree, shared);
        }

        if (has_extra) {
            int2 cmp = -isgreater(extra.xy, origin);
            uint extra_qno = cmp.x << 1 | cmp.y;

            if (extra_qno != qno) {
                uint quarter = node->quarters[extra_qno] = alloc_node(quadtree, shared);
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

    assert(0 && "UNREACHABLE");
}

kernel void run(
    global const point_t *restrict points,
    global node_t *restrict quadtree,
    global shared_t *restrict shared
) {
    printf("RUN GID=%d LID=%d GR=%d BBOX=%.2v4f MD=%d\n",
            get_global_id(0), get_local_id(0), get_group_id(0), shared->bbox, MAX_DEPTH);
    int gid = get_global_id(0);
    insert(quadtree, points[gid], shared);
}
