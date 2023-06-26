# The Spatha Library

(see /include/spatha/block_sparse/spmm)

In order to use the N:M version, keep default ```blockwise_kernel.h``` file in the compilation

To use the baseline kernel, use instead ```blockwise_kernel_baseline.h```

To simulate an ideal scenario where no accesses to column-loc, compile spatha with ```blockwise_kernel_ideal.h``` instead

Finally, in order to evaluate Spatha with 32-bit STS instructions, compile with ```blockwise_kernel_32b.h``` and ```epilogue_32b.h``` (located in /include/spatha/block_sparse/common)