using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using System;
using System.Diagnostics;
using System.IO;

namespace ConsoleApp
{
    unsafe sealed class SomeState : IDisposable
    {
        private int count, defaultBlockCount, defaultThreadsPerBlock, warpSize;
        private SomeBasicType* hostBuffer; // results 
        private const string path = "MyKernels.c";
        private const string methodName = "Multiply";
        CudaDeviceVariable<SomeBasicType> deviceBuffer;
        CudaContext ctx;
        CudaStream defaultStream;
        CudaKernel multiply;
        public SomeState(int deviceId)
        {
            ctx = new CudaContext(deviceId, true);
            var props = ctx.GetDeviceInfo();
            defaultBlockCount = props.MultiProcessorCount * 32;
            defaultThreadsPerBlock = props.MaxThreadsPerBlock;
            warpSize = props.WarpSize;
        }
        internal nvrtcResult LoadKernel(out string log)
        {
            nvrtcResult result;
            using (var rtc = new CudaRuntimeCompiler(File.ReadAllText(path), Path.GetFileName(path)))
            {
                try
                {
                    rtc.Compile(Array.Empty<string>());
                    result = nvrtcResult.Success;
                }
                catch (NVRTCException ex)
                {
                    result = ex.NVRTCError;
                }
                log = rtc.GetLogAsString();

                if (result == nvrtcResult.Success)
                {
                    byte[] ptx = rtc.GetPTX();
                    multiply = ctx.LoadKernelFatBin(ptx, methodName);
                }
            }
            return result;
        }

        public void InitializeData(int count)
        {
            this.count = count;
            IntPtr hostPointer = IntPtr.Zero;
            var res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref hostPointer, count * sizeof(SomeBasicType));
            if (res != CUResult.Success) throw new CudaException(res);
            hostBuffer = (SomeBasicType*)hostPointer;
            deviceBuffer = new CudaDeviceVariable<SomeBasicType>(count);
            for (int i = 0; i < count; i++)
            {
                // we'll just set the key and value to i, so: [{0,0},{1,1},{2,2}...]
                hostBuffer[i].Id = i;
                hostBuffer[i].Value = (uint)i;
            }
            defaultStream = new CudaStream();
            res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(
                deviceBuffer.DevicePointer,
                hostPointer,
                deviceBuffer.SizeInBytes,
                defaultStream.Stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        private static int RoundUp(int value, int blockSize)
        {
            if ((value % blockSize) != 0)
            {
                value += blockSize - (value % blockSize);
            }
            return value;
        }

        internal void MultiplyAsync(int value)
        {
            int threadsPerBlock, blockCount;
            if (count <= defaultThreadsPerBlock)
            {
                blockCount = 1;
                threadsPerBlock = RoundUp(count, warpSize);
            }
            else if (count >= defaultThreadsPerBlock * defaultBlockCount)
            {
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = defaultBlockCount;
            }
            else
            {
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = (count + threadsPerBlock - 1) / threadsPerBlock;
            }
            multiply.BlockDimensions = new ManagedCuda.VectorTypes.dim3(threadsPerBlock, 1, 1);
            multiply.GridDimensions = new ManagedCuda.VectorTypes.dim3(blockCount, 1, 1);

            multiply.RunAsync(defaultStream.Stream, new object[] {
                // note the signature is (N, data, factor)
                count, deviceBuffer.DevicePointer, value
            });
        }

        internal void Synchronize()
        {
            ctx.Synchronize();
        }

        internal void CopyToHost()
        {
            var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(
                new IntPtr(hostBuffer), deviceBuffer.DevicePointer, deviceBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        public SomeBasicType this[int index]
        {
            get
            {
                if (index < 0 || index >= count) throw new IndexOutOfRangeException();
                return hostBuffer[index];
            }
        }

        public void Dispose() => Dispose(true);
        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                GC.SuppressFinalize(this);
            }
            if (hostBuffer != default(SomeBasicType*))
            {
                var tmp = new IntPtr(hostBuffer);
                hostBuffer = default(SomeBasicType*);
                try
                {
                    DriverAPINativeMethods.MemoryManagement.cuMemFreeHost(tmp);
                }
                catch (Exception ex) { Debug.WriteLine(ex.Message); }
            }

            if (disposing)
            {
                Dispose(ref deviceBuffer);
                Dispose(ref defaultStream);
                Dispose(ref ctx);
            }
        }

        private void Dispose<T>(ref T field) where T : class, IDisposable
        {
            if (field != null)
            {
                try { field.Dispose(); } catch (Exception ex) { Debug.WriteLine(ex.Message); }
                field = null;
            }
        }
    }
}
