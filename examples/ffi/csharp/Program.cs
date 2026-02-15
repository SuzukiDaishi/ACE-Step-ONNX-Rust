using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

internal static class Native
{
    [DllImport("acestep_runtime.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr ace_create_context(string config_json);

    [DllImport("acestep_runtime.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void ace_free_context(IntPtr ctx);

    [DllImport("acestep_runtime.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void ace_string_free(IntPtr ptr);

    [DllImport("acestep_runtime.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr ace_last_error();

    [DllImport("acestep_runtime.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int ace_prepare_step_inputs(
        IntPtr ctx,
        string state_json,
        float[] in_tensor_ptr,
        UIntPtr in_tensor_len,
        out IntPtr out_json
    );

    [DllImport("acestep_runtime.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int ace_scheduler_step(
        IntPtr ctx,
        float[] xt_ptr,
        float[] vt_ptr,
        UIntPtr len,
        float dt,
        [Out] float[] out_xt_ptr
    );

    [DllImport("acestep_runtime.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int ace_apply_lm_constraints(
        IntPtr ctx,
        float[] logits_ptr,
        UIntPtr vocab_size,
        [Out] float[] out_masked_logits_ptr
    );
}

public class Program
{
    private static string LastError()
    {
        IntPtr p = Native.ace_last_error();
        if (p == IntPtr.Zero)
        {
            return "unknown";
        }
        try
        {
            return Marshal.PtrToStringUTF8(p) ?? "unknown";
        }
        finally
        {
            Native.ace_string_free(p);
        }
    }

    private static bool NearlyEqual(float a, float b, float eps = 1e-7f)
    {
        return Math.Abs(a - b) <= eps;
    }

    public static int Main(string[] args)
    {
        IntPtr ctx = Native.ace_create_context("{\"seed\":42,\"blocked_token_ids\":[1,3],\"forced_token_id\":2}");
        if (ctx == IntPtr.Zero)
        {
            Console.WriteLine($"create_context failed: {LastError()}");
            return 1;
        }
        try
        {
            float[] inTensor = { 1f, 2f, 3f, 4f };
            int prepRc = Native.ace_prepare_step_inputs(
                ctx,
                "{\"shift\":3.0,\"inference_steps\":8,\"current_step\":0}",
                inTensor,
                (UIntPtr)inTensor.Length,
                out IntPtr outJson
            );
            if (prepRc != 0)
            {
                Console.WriteLine($"ace_prepare_step_inputs failed: {LastError()}");
                return 2;
            }
            string payload = Marshal.PtrToStringUTF8(outJson) ?? "{}";
            Native.ace_string_free(outJson);
            using JsonDocument doc = JsonDocument.Parse(payload);
            float timestep = doc.RootElement.GetProperty("timestep").GetSingle();
            float nextTimestep = doc.RootElement.GetProperty("next_timestep").GetSingle();
            if (!NearlyEqual(timestep, 1.0f) || !NearlyEqual(nextTimestep, 0.669921875f, 1e-6f))
            {
                Console.WriteLine($"prepare mismatch: t={timestep}, next={nextTimestep}");
                return 3;
            }

            float[] xt = { 1f, 1f, 1f, 1f };
            float[] vt = { 0.1f, 0.2f, 0.3f, 0.4f };
            float[] outXt = new float[4];
            int rc = Native.ace_scheduler_step(ctx, xt, vt, (UIntPtr)4, 0.5f, outXt);
            if (rc != 0)
            {
                Console.WriteLine($"ace_scheduler_step failed: {LastError()}");
                return 4;
            }
            float[] expectedXt = { 0.95f, 0.9f, 0.85f, 0.8f };
            for (int i = 0; i < expectedXt.Length; i++)
            {
                if (!NearlyEqual(outXt[i], expectedXt[i]))
                {
                    Console.WriteLine($"scheduler mismatch at {i}: got={outXt[i]} expected={expectedXt[i]}");
                    return 5;
                }
            }

            float[] logits = { 0f, 1f, 2f, 3f, 4f };
            float[] masked = new float[5];
            int lmRc = Native.ace_apply_lm_constraints(ctx, logits, (UIntPtr)logits.Length, masked);
            if (lmRc != 0)
            {
                Console.WriteLine($"ace_apply_lm_constraints failed: {LastError()}");
                return 6;
            }
            if (!NearlyEqual(masked[2], 2f))
            {
                Console.WriteLine($"forced token mismatch: {masked[2]}");
                return 7;
            }
            for (int i = 0; i < masked.Length; i++)
            {
                if (i == 2)
                {
                    continue;
                }
                if (masked[i] > -1e29f)
                {
                    Console.WriteLine($"mask mismatch at {i}: {masked[i]}");
                    return 8;
                }
            }
            Console.WriteLine("csharp ffi regression: PASS");
            return 0;
        }
        finally
        {
            Native.ace_free_context(ctx);
        }
    }
}
