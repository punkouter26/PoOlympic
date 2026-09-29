using System;
using Unity.InferenceEngine;

namespace PoOlympic
{
    /// <summary>
    /// Batch-1 ONNX policy on the Inference Engine CPU backend. Outputs are exactly the graph's `ctrl` (final PD
    /// targets, already clipped) and `action_raw` (fed back as last_action) — C# does no action math.
    /// </summary>
    public sealed class PolicyBrain : IDisposable
    {
        readonly Worker _worker;
        readonly Tensor<float> _input;
        readonly int _obsDim;

        public PolicyBrain(ModelAsset asset, int obsDim)
        {
            _obsDim = obsDim;
            _worker = new Worker(ModelLoader.Load(asset), BackendType.CPU);
            _input = new Tensor<float>(new TensorShape(1, obsDim));
        }

        public void Run(float[] obs, float[] ctrl, float[] actionRaw)
        {
            if (obs.Length != _obsDim) throw new ArgumentException("obs length");
            _input.Upload(obs);
            _worker.Schedule(_input);
            using var c = (_worker.PeekOutput("ctrl") as Tensor<float>).ReadbackAndClone();
            using var a = (_worker.PeekOutput("action_raw") as Tensor<float>).ReadbackAndClone();
            var cArr = c.DownloadToArray();
            var aArr = a.DownloadToArray();
            Array.Copy(cArr, ctrl, ctrl.Length);
            Array.Copy(aArr, actionRaw, actionRaw.Length);
        }

        public void Dispose()
        {
            _input?.Dispose();
            _worker?.Dispose();
        }
    }

    /// <summary>
    /// The brain's PPO critic (tools/export_critic.py → &lt;brain&gt;.critic.onnx): obs → value, the return the policy
    /// expects from here on. Read-only telemetry for the broadcast (brain confidence) — never feeds back into ctrl.
    /// </summary>
    public sealed class PolicyCritic : IDisposable
    {
        readonly Worker _worker;
        readonly Tensor<float> _input;
        public readonly int ObsDim;

        public PolicyCritic(ModelAsset asset)
        {
            var model = ModelLoader.Load(asset);
            ObsDim = model.inputs[0].shape.Get(1);
            _worker = new Worker(model, BackendType.CPU);
            _input = new Tensor<float>(new TensorShape(1, ObsDim));
        }

        public float Run(float[] obs)
        {
            _input.Upload(obs);
            _worker.Schedule(_input);
            using var v = (_worker.PeekOutput("value") as Tensor<float>).ReadbackAndClone();
            return v.DownloadToArray()[0];
        }

        public void Dispose()
        {
            _input?.Dispose();
            _worker?.Dispose();
        }
    }
}
