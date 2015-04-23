/*
 * Copyright 2008-2013 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For more information on Heaton Research copyrights, licenses
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
//Source http://wiki.lwjgl.org/wiki/OpenCL_in_LWJGL
package com.jeffheaton.opencl;

import org.lwjgl.BufferUtils;
import org.lwjgl.LWJGLException;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.*;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

import static org.lwjgl.opencl.CL10.*;

public class HelloOpenCL {
    public static void main(String... args) throws Exception {
        displayInfo();
        System.out.println("--------------------------------------------");
        if (args.length > 0 && args[0].contains("once")) {
            {
                System.out.println("Starting GPU benchmark");
                long totalTime = benchmark();
                System.out.println("Total execution time in ns:" + totalTime);
                System.out.println("Total execution time in ms:" + TimeUnit.MILLISECONDS.convert(totalTime, TimeUnit.NANOSECONDS));
            }
            System.out.println("--------------------------------------------");
            {
                System.out.println("Starting CPU benchmark");
                long totalTime = benchmark("CPU");
                System.out.println("Total execution time in ns:" + totalTime);
                System.out.println("Total execution time in ms:" + TimeUnit.MILLISECONDS.convert(totalTime, TimeUnit.NANOSECONDS));
            }
        } else {
            System.out.println("Starting GPU benchmark");
            int benchmarks = 1000;
            int pauseBetweenBenchmarks=100;
            double[] gpuBenchmarkResults = new double[benchmarks];
            for (int i = 0; i < benchmarks; i++) {
                long res = benchmark();
                gpuBenchmarkResults[i] = ((double)res)/1000000d; //convert to ms
                Thread.sleep(pauseBetweenBenchmarks);
            }

            System.out.println("--------------------------------------------");
            System.out.println("Starting CPU benchmark");
            double[] cpuBenchmarkResults = new double[benchmarks];
            for (int i = 0; i < benchmarks; i++) {
                long res = benchmark("CPU");
                cpuBenchmarkResults[i] = ((double)res)/1000000d; //convert to ms
                Thread.sleep(pauseBetweenBenchmarks);
            }

            //Print results

            System.out.println("#GPU Result:#");
            for (int i = 0; i < gpuBenchmarkResults.length; i++) {
                if(i==0) System.out.print("[");
                System.out.print(gpuBenchmarkResults[i]);
                if(i!=gpuBenchmarkResults.length-1) System.out.print(",");
                else System.out.print("]");
            }
            System.out.println();
            System.out.println("#CPU Result:#");
            for (int i = 0; i < cpuBenchmarkResults.length; i++) {
                if(i==0) System.out.print("[");
                System.out.print(cpuBenchmarkResults[i]);
                if(i!=cpuBenchmarkResults.length-1) System.out.print(",");
                else System.out.print("]");
            }
            System.out.println();
        }

    }

    /**
     * @param args "CPU" or nothing
     * @return
     * @throws Exception
     */
    public static long benchmark(String... args) throws Exception {
        final FloatBuffer a = UtilCL.toFloatBuffer(geberateFloatData(100000, 1));
        final FloatBuffer b = UtilCL.toFloatBuffer(geberateFloatData(100000, 94673));
        final FloatBuffer answer = BufferUtils.createFloatBuffer(a.capacity());

        // Initialize OpenCL and create a context and command queue
        CL.create();

        CLPlatform platform = null;
        for (int platformIndex = 0; platformIndex < CLPlatform.getPlatforms().size(); platformIndex++) {
            platform = CLPlatform.getPlatforms().get(platformIndex);
            List<CLDevice> devices;
            if (args.length == 1 && args[0].equalsIgnoreCase("cpu")) devices = platform.getDevices(CL_DEVICE_TYPE_CPU);
            else devices = platform.getDevices(CL_DEVICE_TYPE_GPU);
            if (devices == null) continue;
            if (devices.size() >= 1) break;
        }

        List<CLDevice> devices;
        if (args.length == 1 && args[0].equalsIgnoreCase("cpu")) devices = platform.getDevices(CL_DEVICE_TYPE_CPU);
        else devices = platform.getDevices(CL_DEVICE_TYPE_GPU);
        System.out.println("Running Benchmark on: ");
        displayDeviceInfo(devices.get(0), 0);
        CLContext context = CLContext.create(platform, devices, null, null, null);
        CLCommandQueue queue = clCreateCommandQueue(context, devices.get(0), CL_QUEUE_PROFILING_ENABLE, null);

        // Allocate memory for our two input buffers and our result buffer
        CLMem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a, null);
        clEnqueueWriteBuffer(queue, aMem, 1, 0, a, null, null);
        CLMem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b, null);
        clEnqueueWriteBuffer(queue, bMem, 1, 0, b, null, null);
        CLMem answerMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, answer, null);
        clFinish(queue);

        // Load the source from a resource file
        String source = UtilCL.getResourceAsString("cl/calc.txt");

        // Create our program and kernel
        CLProgram program = clCreateProgramWithSource(context, source, null);
        int error = (clBuildProgram(program, devices.get(0), "", null));
        String compOut = program.getBuildInfoString(devices.get(0), CL_PROGRAM_BUILD_LOG);

        if (compOut != null) {
            System.out.println(compOut);
        } else {
            System.out.println("No compiler output available.");
        }

        Util.checkCLError(error);
        // calc has to match a kernel method name in the OpenCL source
        CLKernel kernel = clCreateKernel(program, "calc", null);

        // Parameters
        PointerBuffer kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
        kernel1DGlobalWorkSize.put(0, a.capacity());
        kernel.setArg(0, aMem);
        kernel.setArg(1, bMem);
        kernel.setArg(2, answerMem);

        //Enqueue for execution
        long startTime = System.nanoTime();
        clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);

        // Read the results memory back into our result buffer
        clEnqueueReadBuffer(queue, answerMem, 1, 0, answer, null, null);
        clFinish(queue);
        long endTime = System.nanoTime();

        long totalTime = endTime - startTime;
        // Print the result memory
//        print(a);
//        System.out.println("%");
//        print(b);
//        System.out.println("=");
//        print(answer);
        System.out.println("Cleaning up...");
        // Clean up OpenCL resources
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(aMem);
        clReleaseMemObject(bMem);
        clReleaseMemObject(answerMem);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        CL.destroy();

//        if (args.length == 0) {
//            System.out.println("Starting CPU benchmark");
////            benchmark("CPU");
//        }
        return totalTime;
    }

    // Data buffers to store the input and result data in

    public static void displayInfo() throws LWJGLException {
        CL.create();
        for (int platformIndex = 0; platformIndex < CLPlatform.getPlatforms().size(); platformIndex++) {
            CLPlatform platform = CLPlatform.getPlatforms().get(platformIndex);
            System.out.println("Platform #" + platformIndex + ":" + platform.getInfoString(CL_PLATFORM_NAME));
            List<CLDevice> devices = platform.getDevices(CL_DEVICE_TYPE_ALL);
            if (devices == null) continue;
            for (int deviceIndex = 0; deviceIndex < devices.size(); deviceIndex++) {
                CLDevice device = devices.get(deviceIndex);
                displayDeviceInfo(device, deviceIndex);
            }
        }
        CL.destroy();
    }

    public static void displayDeviceInfo(CLDevice device, int deviceIndex) {
        System.out.printf(Locale.ENGLISH, "Device #%d(%s):%s\n",
                deviceIndex,
                UtilCL.getDeviceType(device.getInfoInt(CL_DEVICE_TYPE)),
                device.getInfoString(CL_DEVICE_NAME));
        System.out.printf(Locale.ENGLISH, "\tCompute Units: %d @ %d mghtz\n",
                device.getInfoInt(CL_DEVICE_MAX_COMPUTE_UNITS), device.getInfoInt(CL_DEVICE_MAX_CLOCK_FREQUENCY));
        System.out.printf(Locale.ENGLISH, "\tLocal memory: %s\n",
                UtilCL.formatMemory(device.getInfoLong(CL_DEVICE_LOCAL_MEM_SIZE)));
        System.out.printf(Locale.ENGLISH, "\tGlobal memory: %s\n",
                UtilCL.formatMemory(device.getInfoLong(CL_DEVICE_GLOBAL_MEM_SIZE)));
        System.out.println();
    }

    private static float[] geberateFloatData(int i, int modifier) {
        float[] generated = new float[i];
        for (int j = 0; j < i; j++) {
            generated[j] = modifier * (j % 10);
        }
        return generated;
    }
}