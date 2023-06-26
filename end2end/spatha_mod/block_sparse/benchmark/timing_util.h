#include <cuda_runtime.h>

struct GpuTimer
{
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    GpuTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    ~GpuTimer() 
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start()
    {
        cudaEventRecord(startEvent, 0);
    }

    void stop()
    {
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
    }

    float elapsed_msecs()
    {
        float elapsed;
        cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
        return elapsed;
    }
};
