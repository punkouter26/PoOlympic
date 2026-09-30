using System;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Rung S showcase: cycles a contract v4 PolicyRunner through the five stance skills (stand → squat → flamingo → march →
    /// torso aim → reach, both sides) and labels the current one. Commands are mid-range values of the training ranges
    /// (training/poolympic/tasks/skill_mdp.py SKILL_RANGES). Untick <see cref="cycle"/> to drive PolicyRunner.skill by hand
    /// in the Inspector.
    /// </summary>
    public class StanceSkillDemo : MonoBehaviour
    {
        public PolicyRunner runner;
        [Tooltip("Step through the skills automatically; off = edit PolicyRunner.skill in the Inspector.")]
        public bool cycle = true;
        [Tooltip("Seconds per skill (stand-still breaks are half this).")]
        public float secondsPerSkill = 6f;

        [Serializable]
        public struct Step
        {
            public string label;
            public SkillCommand skill;
            public bool rest;
        }

        public Step[] steps;
        int _i = -1;
        float _until;
        GUIStyle _style;

        void Reset() => steps = DefaultSteps();

        public static Step[] DefaultSteps()
        {
            Step S(string label, SkillCommand s, bool rest = false) => new() { label = label, skill = s, rest = rest };
            var stand = new SkillCommand();
            return new[]
            {
                S("Stand", stand, true),
                S("E3 Deep Squat  (pelvis −0.30 m)", new SkillCommand { pelvisHeight = -0.30f }),
                S("Stand", stand, true),
                S("E6 Flamingo  (left foot up)", new SkillCommand { liftFoot = SkillCommand.Foot.Left }),
                S("Stand", stand, true),
                S("E7 Cadence March  (1.3 Hz, knee 0.20 m)", new SkillCommand { marchHz = 1.3f, kneeLift = 0.20f }),
                S("Stand", stand, true),
                S("E2 Torso Archer  (yaw +30°, pitch +15°)", new SkillCommand { torsoYaw = 0.52f, torsoPitch = 0.26f }),
                S("E2 Torso Archer  (yaw −30°, pitch −10°)", new SkillCommand { torsoYaw = -0.52f, torsoPitch = -0.17f }),
                S("Stand", stand, true),
                S("E4 Javelin Reach  (right arm)", new SkillCommand { arm = 1, handTarget = new Vector3(0.437f, -0.484f, 0.742f) }),
                S("E4 Javelin Reach  (left arm)", new SkillCommand { arm = -1, handTarget = new Vector3(0.437f, 0.484f, 0.742f) }),
                S("E3 Deep Squat  (pelvis −0.40 m)", new SkillCommand { pelvisHeight = -0.40f }),
                S("E6 Flamingo  (right foot up)", new SkillCommand { liftFoot = SkillCommand.Foot.Right }),
            };
        }

        void Update()
        {
            if (!cycle || runner == null || steps == null || steps.Length == 0) return;
            if (_i >= 0 && Time.time < _until) return;
            _i = (_i + 1) % steps.Length;
            runner.skill = steps[_i].skill;
            runner.command = Vector3.zero;
            _until = Time.time + (steps[_i].rest ? secondsPerSkill * 0.5f : secondsPerSkill);
        }

        void OnGUI()
        {
            if (runner == null) return;
            // sized by the narrow side and wrapped: the phone game view is 960 wide (a height-based size clipped the label)
            _style ??= new GUIStyle(GUI.skin.label)
            {
                fontSize = Mathf.RoundToInt(Mathf.Min(Screen.width / 26f, Screen.height / 32f)), fontStyle = FontStyle.Bold,
                wordWrap = true,
            };
            string label = cycle && _i >= 0 ? steps[_i].label : "Manual (PolicyRunner.skill)";
            string brain = runner.brain != null ? runner.brain.name : "no brain";
            float w = Screen.width - 40;
            float h = _style.CalcHeight(new GUIContent(label), w);
            var r = new Rect(20, Screen.height * 0.12f, w, h);
            GUI.color = Color.black;
            GUI.Label(new Rect(r.x + 2, r.y + 2, r.width, r.height), label, _style);
            GUI.color = Color.white;
            GUI.Label(r, label, _style);
            var small = new GUIStyle(_style) { fontSize = _style.fontSize / 2, fontStyle = FontStyle.Normal };
            GUI.Label(new Rect(r.x, r.y + h, w, small.fontSize * 1.6f), brain, small);
        }
    }
}
