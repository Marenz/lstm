import std.stdio;

import mir.ndslice;

auto sigmoid ( T ) ( T x )
{
    import std.math : exp;

    return 1.0 / (1.0 + exp(-x));
}

// Create a two dimensional array filled with random values
auto randomSlice ( Lengths... ) ( double min, double max, Lengths lengths )
{
    import std.random;

    auto matrix = slice!double(lengths);

    matrix.ndEach!((ref a) => a = uniform(min, max));

    return matrix;
}

alias Matrix2d = Slice!(2, double*);
alias Vector = Slice!(1, double*);

class LstmParam
{
    long mem_cell_ct;
    long x_dim;

    // Weight matrices
    Matrix2d wg, wi, wf, wo;

    // Bias terms
    Vector bg, bi, bf, bo;

    // diffs
    Matrix2d wg_diff, wi_diff, wf_diff, wo_diff;
    Vector bg_diff, bi_diff, bf_diff, bo_diff;

    this ( long mem_cell_ct, long x_dim )
    {
        this.mem_cell_ct = mem_cell_ct;
        this.x_dim = x_dim;

        auto concat_len = x_dim + mem_cell_ct;

        this.wg = randomSlice(-0.1, 0.1, mem_cell_ct, concat_len);
        this.wi = randomSlice(-0.1, 0.1, mem_cell_ct, concat_len);
        this.wf = randomSlice(-0.1, 0.1, mem_cell_ct, concat_len);
        this.wo = randomSlice(-0.1, 0.1, mem_cell_ct, concat_len);

        this.bg = randomSlice(-0.1, 0.1, mem_cell_ct);
        this.bi = randomSlice(-0.1, 0.1, mem_cell_ct);
        this.bf = randomSlice(-0.1, 0.1, mem_cell_ct);
        this.bo = randomSlice(-0.1, 0.1, mem_cell_ct);

        this.wg_diff = slice([mem_cell_ct, concat_len], cast(double)0.0);
        this.wi_diff = slice([mem_cell_ct, concat_len], cast(double)0.0);
        this.wf_diff = slice([mem_cell_ct, concat_len], cast(double)0.0);
        this.wo_diff = slice([mem_cell_ct, concat_len], cast(double)0.0);


        this.bg_diff = slice([mem_cell_ct], cast(double)0.0);
        this.bi_diff = slice([mem_cell_ct], cast(double)0.0);
        this.bf_diff = slice([mem_cell_ct], cast(double)0.0);
        this.bo_diff = slice([mem_cell_ct], cast(double)0.0);
    }

    void applyDiff ( double lr = 1.0L )
    {
        void apply (S) ( ref S a, ref S diff )
        {
            import mir.ndslice.slice : assumeSameStructure;
            import mir.ndslice.algorithm;

            ndEach!((ref a) { a *= lr; })(diff);

            auto zip = assumeSameStructure!("a", "b")(a, diff);

            zip.ndEach!(z => z.a -= z.b)();
        }

        apply(this.wg, this.wg_diff);
        apply(this.wi, this.wi_diff);
        apply(this.wf, this.wf_diff);
        apply(this.wo, this.wo_diff);

        apply(this.bg, this.bg_diff);
        apply(this.bi, this.bi_diff);
        apply(this.bf, this.bf_diff);
        apply(this.bo, this.bo_diff);


        this.wg_diff[] = cast(double)0.0;
        this.wi_diff[] = cast(double)0.0;
        this.wf_diff[] = cast(double)0.0;
        this.wo_diff[] = cast(double)0.0;

        this.bg_diff[] = 0.0;
        this.bi_diff[] = 0.0;
        this.bf_diff[] = 0.0;
        this.bo_diff[] = 0.0;

    }
}

class LstmState
{
    Vector g, i, f, o ,s ,h,
           bottom_diff_h, bottom_diff_s, bottom_diff_x;

    this ( long mem_cell_ct, long x_dim )
    {
        this.g = slice([mem_cell_ct], cast(double)0.0);
        this.i = slice([mem_cell_ct], 0.0);
        this.f = slice([mem_cell_ct], 0.0);
        this.o = slice([mem_cell_ct], 0.0);
        this.s = slice([mem_cell_ct], 0.0);
        this.h = slice([mem_cell_ct], 0.0);

        this.bottom_diff_h = slice([this.h.length], 0.0);
        this.bottom_diff_s = slice([this.s.length], 0.0);
        this.bottom_diff_x = slice([x_dim], 0.0);
    }
}


class LstmNode
{
    LstmParam param;
    LstmState state;

    Vector x, xc;

    Vector s_prev, h_prev;

    this ( LstmParam param, LstmState state )
    {
        this.param = param;
        this.state = state;
    }

    void bottomDataIs ( Vector x, Vector s_prev = Vector(),
                        Vector h_prev = Vector() )
    {
        // If this is the first lstm node in the network
        if (s_prev.length == 0) s_prev = slice([this.state.s.length], 0.0);
        if (h_prev.length == 0) h_prev = slice([this.state.h.length], 0.0);

        // save data for use in backprop
        this.s_prev = s_prev;
        this.h_prev = h_prev;

        import std.range;

        double[] merged = x.ptr[0 .. x.elementsCount] ~
                        s_prev.ptr[0 ..  s_prev.elementsCount];

        // concatenate x(t) and h(t-1)
        auto xc = merged.sliced;
        assert(xc.length == merged.length);

        import mir.glas.l2;
        import mir.glas.common;
        import mir.ndslice.slice : assumeSameStructure;
        import std.math : tanh;

        GlasContext glas;

        // assigns result of func(pw.dot(xc) + pb) to state
        void apply ( alias func ) ( ref Vector state, Matrix2d pw, Vector pb )
        {
            state = pw.dot(xc).perElement!add(pb).ndMap!(a =>
                    cast(double)func(a)).slice;

            //gemv!(double, double, double)(&glas, 1.0L, pw, xc, 0.0, state);
            //auto zip = assumeSameStructure!("a", "b")(state, pb);
            //zip.ndEach!(z => z.a = func(z.a + z.b))();
        }

        apply!tanh(this.state.g, this.param.wg, this.param.bg);
        apply!sigmoid(this.state.i, this.param.wi, this.param.bi);
        apply!sigmoid(this.state.f, this.param.wf, this.param.bf);
        apply!sigmoid(this.state.o, this.param.wo, this.param.bo);


        auto s_zip = assumeSameStructure!("g", "i", "s_prev", "f")(
                this.state.g, this.state.i, this.s_prev, this.state.f);
        this.state.s = s_zip.ndMap!(z => z.g * z.i + z.s_prev * z.f).slice;

        this.state.h = this.state.s.perElement!mul(this.state.o);

        this.x = x;
        this.xc = xc;
    }

    void topDiffIs ( Vector top_diff_h, Vector top_diff_s )
    {
        // notice that top_diff_s is carried along the constant error carousel
        auto ds  = this.state.o.perElement!mul(top_diff_h)
                               .perElement!add(top_diff_s);
        auto _do = this.state.s.perElement!mul(top_diff_h);
        auto di  = this.state.g.perElement!mul(ds);
        auto dg  = this.state.i.perElement!mul(ds);
        auto df  = this.s_prev.perElement!mul(ds);

        // diffs w.r.t. vector inside sigma / tanh function
        auto di_input = 1.0L.perElement!sub(this.state.i)
                            .perElement!mul(this.state.i)
                            .perElement!mul(di);

        auto df_input = 1.0L.perElement!sub(this.state.f)
                            .perElement!mul(this.state.f)
                            .perElement!mul(df);

        auto do_input = 1.0L.perElement!sub(this.state.o)
                            .perElement!mul(this.state.o)
                            .perElement!mul(_do);

        auto dg_input = 1.0L.perElement!sub(this.state.g
                                            .perElement!mul(this.state.g))
                            .perElement!mul(dg);

        // diffs w.r.t. inputs
        this.param.wi_diff = this.param.wi_diff
                                    .perElement!add(di_input.outer(this.xc));
        this.param.wf_diff = this.param.wf_diff
                                    .perElement!add(df_input.outer(this.xc));
        this.param.wo_diff = this.param.wo_diff
                                    .perElement!add(do_input.outer(this.xc));
        this.param.wg_diff = this.param.wg_diff
                                    .perElement!add(dg_input.outer(this.xc));

        this.param.bi_diff = this.param.bi_diff.perElement!add(di_input);
        this.param.bf_diff = this.param.bi_diff.perElement!add(df_input);
        this.param.bo_diff = this.param.bi_diff.perElement!add(do_input);
        this.param.bg_diff = this.param.bi_diff.perElement!add(dg_input);

        // compute bottom diff
        //typeof(this.xc) dxc; dxc[] = 0.0;
        auto dxc = slice!double(this.xc.shape, 0.0);

        import mir.ndslice.iteration;

        dxc = dxc.perElement!add(this.param.wi.transposed.dot(di_input));
        dxc = dxc.perElement!add(this.param.wf.transposed.dot(df_input));
        dxc = dxc.perElement!add(this.param.wo.transposed.dot(do_input));
        dxc = dxc.perElement!add(this.param.wg.transposed.dot(dg_input));

        // save bottom diffs
        this.state.bottom_diff_s = ds.perElement!mul(this.state.f);
        this.state.bottom_diff_x = dxc.byElement[0 .. this.param.x_dim];
        this.state.bottom_diff_h = dxc.byElement[this.param.x_dim .. $];
    }
}

class LstmNetwork
{
    interface LossLayer
    {
        double loss ( Vector pred, double label );
        Vector bottomDiff ( Vector pred, double label );
    }

    LstmParam param;

    LstmNode[] node_list;
    // Input vector
    Matrix2d x_list;

    this ( LstmParam param )
    {
        this.param = param;
    }


    /* Updates diffs by setting target sequence with corresponding loss layer.
       Will *NOT* update parameters. To update parameters,
       call this.lstm_param.applyDiff()
       */
    double yListIs ( Vector y_list, LossLayer loss_layer )
    {
        assert(x_list.length == y_list.length);
        assert(x_list.length > 0);

        long idx = this.x_list.length - 1;

        // first node only gets diffs from label …
        auto loss = loss_layer.loss(this.node_list[idx].state.h, y_list[idx]);
        auto diff_h = loss_layer.bottomDiff(this.node_list[idx].state.h,
                                            y_list[idx]);

        // here s is not affecting loss due to h(t+1), hence we set equal to
        // zero
        auto diff_s = slice([this.param.mem_cell_ct], 0.0);
        this.node_list[idx].topDiffIs(diff_h, diff_s);

        idx -= 1;

        // ... following nodes also get diffs from next nodes, hence we add
        // diffs to diff_h. We also propagate error along constant error
        // carousel using diff_s

        while (idx >= 0)
        {
            loss += loss_layer.loss(
                        this.node_list[idx].state.h, y_list[idx]);
            diff_h = loss_layer.bottomDiff(this.node_list[idx].state.h,
                                           y_list[idx]);
            diff_h = diff_h.perElement!add(this.node_list[idx+1].state.bottom_diff_h);
            diff_s = this.node_list[idx+1].state.bottom_diff_s;
            this.node_list[idx].topDiffIs(diff_h, diff_s);

            idx -= 1;
        }

        return loss;
    }

    void xListClear ( )
    {
        this.x_list = slice!double([0, 0]);
    }

    void xListAdd ( Vector x )
    {
        auto new_matrix = slice!double([this.x_list.length + 1, x.length]);

        //writefln("matrix created, lens: %s", new_matrix.shape());
        //writefln("vec shape: %s", x.shape());
        //writefln("partial matrix shape: %s", new_matrix[0..$-1].shape);

        if (this.x_list.length > 0)
            new_matrix[0 .. $-1][] = this.x_list;

        new_matrix[$-1 .. $][] = x;

        this.x_list = new_matrix;

        if (this.x_list.length > this.node_list.length)
        {
            // need to add new lstm node, create new state mem
            auto state = new LstmState(this.param.mem_cell_ct,
                                       this.param.x_dim);
            this.node_list ~= new LstmNode(this.param, state);
        }

        // get index of most recent x input;
        auto idx = this.x_list.length - 1;

        if (idx == 0)
        {
            // no recurrent inputs yet
            this.node_list[idx].bottomDataIs(x);
        }
        else
        {
            auto s_prev = this.node_list[idx - 1].state.s;
            auto h_prev = this.node_list[idx - 1].state.h;
            this.node_list[idx].bottomDataIs(x, s_prev, h_prev);
        }
    }
}

auto mul ( T ) ( T a, T b ) { return a * b; }
auto add ( T ) ( T a, T b ) { return a + b; }
auto sub ( T ) ( T a, T b ) { return a - b; }


auto dot ( Matrix2d mat, Vector vec )
{
    import mir.glas.l2;
    import mir.glas.common;

    GlasContext glas;

    Vector result = slice!double(mat.length);


    gemv!(double, double, double)(&glas, 1.0L, mat, vec, 0.0, result);

    return result;
}

// Per forms per-element operation of the constant on the vector
auto perElement ( alias op ) ( double a, Vector b )
{
    return b.ndMap!(x => op(a, x)).slice;
}

// Per forms per-element operation on the two vectors
auto perElement ( alias op, T ) ( T a, T b )
{
    auto zip = assumeSameStructure!("a", "b")(a, b);

    return zip.ndMap!(z => op(z.a, z.b)).slice;
}

// Calculates the outer product of a and b
auto outer ( Vector a, Vector b )
{
    auto result = slice!double(a.length, b.length);

    size_t idx_a, idx_b;

    while (idx_a < a.length)
    {
        while (idx_b < b.length)
        {
            result[idx_a, idx_b] = a[idx_a] * b[idx_b];
            idx_b++;
        }
        idx_a++;
        idx_b=0;
    }

    return result;
}


// Computes square loss with first element of hidden layer array
class ToyLossLayer : LstmNetwork.LossLayer
{
    override double loss ( Vector pred, double label )
    {
        return (pred[0] - label) * (pred[0] - label);
    }
    override Vector bottomDiff ( Vector pred, double label )
    {
        auto diff = slice!double([pred.length], 0.0);
        diff[0] = 2 * (pred[0] - label);
        return diff;
    }
}


void main()
{
    // learns to repeat simple sequence from random inputs
    auto loss_layer = new ToyLossLayer;

    auto mem_cell_ct = 100;
    auto x_dim = 50;
    auto concat_len = x_dim + mem_cell_ct;
    auto param = new LstmParam(mem_cell_ct, x_dim);
    auto net = new LstmNetwork(param);
    auto y_list = slice!double([4]);
    y_list[] = [-0.5, 0.2, 0.1, -0.5];

    auto input_val_arr = randomSlice(-1.0L, 1.0L, y_list.length, x_dim);

    for ( size_t cur_iter = 0; cur_iter < 100; cur_iter++)
    {
        writefln("cur iter: %s", cur_iter);

        for (size_t ind = 0; ind < y_list.length; ind++)
        {
            net.xListAdd(input_val_arr[ind]);
            writefln("y_pred[%s] : %s", ind, net.node_list[ind].state.h[0]);
        }

        auto loss = net.yListIs(y_list, loss_layer);
        writefln("loss: %s", loss);
        param.applyDiff(0.1L);
        net.xListClear();
    }
}
