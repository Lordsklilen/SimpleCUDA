
using ManagedCuda;
using System;

namespace ConsoleApp
{
    class Program
    {
        static int Main()
        {
            try
            {
                int deviceCount = CudaContext.GetDeviceCount();
                if (deviceCount == 0)
                {
                    Console.Error.WriteLine("No CUDA devices detected. Sad face.");
                    return -1;
                }
                Console.WriteLine($"{deviceCount} CUDA devices detected (first will be used)");
                for (int i = 0; i < deviceCount; i++)
                {
                    Console.WriteLine($"{i}: {CudaContext.GetDeviceName(i)}");
                }
                const int Count = 1024 * 1024;
                using (var state = new SomeState(deviceId: 0))
                {
                    Console.WriteLine("Initializing kernel...");
                    var compileResult = state.LoadKernel(out string log);
                    if (compileResult != ManagedCuda.NVRTC.nvrtcResult.Success)
                    {
                        Console.Error.WriteLine(compileResult);
                        Console.Error.WriteLine(log);
                        return -1;
                    }
                    Console.WriteLine(log);
                    Console.WriteLine("Initializing data...");
                    state.InitializeData(Count);

                    Console.WriteLine("Running kernel...");
                    for (int i = 0; i < 8; i++)
                    {
                        state.MultiplyAsync(2);
                    }
                    Console.WriteLine("Copying data back...");
                    state.CopyToHost();
                    Console.WriteLine("Waiting for completion...");
                    state.Synchronize();
                    Console.WriteLine("all done; showing some results");
                    var random = new Random(123456);
                    for (int i = 0; i < 20; i++)
                    {
                        var record = state[random.Next(Count)];
                        Console.WriteLine($"{i}: {nameof(record.Id)}={record.Id}, {nameof(record.Value)}={record.Value}");
                    }
                    Console.WriteLine("Cleaning up...");
                }
                Console.WriteLine("All done; have a nice day");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
                Console.ReadKey();
                return -1;
            }
            Console.ReadKey();
            return 0;
        }
    }
}
