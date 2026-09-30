// Living crowd (GFX idea 3). The 1.0 m crowd cards on the rows (build_showcase.py crowd_cards, UV v 0.5 at the tread ..
// 1 at the top) sample two textures of the same 64 seats: _BaseMap every fan seated, _CheerMap the same fan cheering
// with arms up (Art/Stadium/Crowd/*.png, V clamped: one atlas bled its halves together at low mips). Per seat (64 seats per u) a hash picks
// seated / cheering and a bob height, driven by globals that CrowdDirector sets from the heat:
//   _PoCrowdExcite  0..1  tension: bob amplitude + share of fans on their feet
//   _PoCrowdCheer   0..1  burst (result, world record, a save): most of the stand cheers
//   _PoCrowdWave    0..1  Mexican wave running round the bowl
//   _PoCrowdFlash   0..1  press / phone camera flashes (emissive seats)
// Lighting = URP Simple Lit (baked lightmaps, main + additional lights, fog); only the surface is custom.
Shader "PoOlympic/Crowd"
{
    Properties
    {
        [MainTexture] _BaseMap("Seated fans (V clamped)", 2D) = "white" {}
        _CheerMap("Cheering fans (same seats, V clamped)", 2D) = "white" {}
        [MainColor] _BaseColor("Tint", Color) = (1, 1, 1, 1)
        _Seats("Seats per UV unit", Float) = 64
        _Bob("Bob height (fraction of a row)", Range(0, 0.3)) = 0.12
        _WaveSpeed("Wave speed (rad/s)", Float) = 0.9
        [HideInInspector] _SpecColor("Specular", Color) = (0.1, 0.1, 0.1, 1)
        [HideInInspector] _EmissionColor("Emission", Color) = (0, 0, 0, 1)
        [HideInInspector] _Cutoff("Cutoff", Float) = 0.5
        [HideInInspector] _Surface("Surface", Float) = 0
    }

    SubShader
    {
        Tags { "RenderType" = "TransparentCutout" "RenderPipeline" = "UniversalPipeline" "UniversalMaterialType" = "SimpleLit" "Queue" = "AlphaTest" }

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }
            Cull Back
            ZWrite On
            AlphaToMask Off

            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex LitPassVertexSimple
            #pragma fragment CrowdFragment

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile _ EVALUATE_SH_MIXED EVALUATE_SH_VERTEX
            #pragma multi_compile _ LIGHTMAP_SHADOW_MIXING
            #pragma multi_compile _ SHADOWS_SHADOWMASK
            #pragma multi_compile _ _CLUSTER_LIGHT_LOOP
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fragment _ _SCREEN_SPACE_OCCLUSION
            #include_with_pragmas "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Fog.hlsl"
            #pragma multi_compile _ DIRLIGHTMAP_COMBINED
            #pragma multi_compile _ LIGHTMAP_ON
            #pragma multi_compile _ USE_LEGACY_LIGHTMAPS
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/Shaders/SimpleLitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/SimpleLitForwardPass.hlsl"

            float _Seats, _Bob, _WaveSpeed;
            TEXTURE2D(_CheerMap); SAMPLER(sampler_CheerMap);
            float _PoCrowdExcite, _PoCrowdCheer, _PoCrowdWave, _PoCrowdFlash;

            float Hash(float2 p)
            {
                p = frac(p * float2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return frac(p.x * p.y);
            }

            void CrowdSurface(Varyings input, out SurfaceData s, out float flash)
            {
                float2 uv = input.uv;                                  // v 0.5 (bottom of the row) .. 1 (top) = seated half
                float seat = floor(uv.x * _Seats);
                float baseY = input.positionWS.y - (uv.y - 0.5) * 2.0;   // the card's foot (cards are 1.0 m tall)
                float row = floor(baseY / 0.4 + 0.5);                     // rows rise 0.4 m
                float h = Hash(float2(seat, row));
                float t = _Time.y;

                float excite = saturate(_PoCrowdExcite);
                float standP = saturate(excite * 0.35 + _PoCrowdCheer * 0.9);
                float slot = floor(t * (0.6 + h * 0.8) + h * 7.0);       // each fan re-decides every ~1-2 s
                bool cheer = Hash(float2(seat + slot * 13.1, row + 3.7)) < standP;
                float ang = atan2(input.positionWS.z, input.positionWS.x);
                float wave = _PoCrowdWave * smoothstep(0.86, 0.98, sin(ang - t * _WaveSpeed));
                cheer = cheer || wave > 0.5;

                float amp = _Bob * saturate(excite * 0.6 + _PoCrowdCheer + wave) * (0.5 + h);
                float bob = amp * (0.5 + 0.5 * sin(t * (7.0 + h * 5.0) + h * 6.2832)) + wave * _Bob;
                float2 card = float2(uv.x, (uv.y - 0.5) * 2.0);         // 0 = the tread .. 1 = top of the card
                float local = saturate(card.y - bob);                   // − bob = the fan rises
                float2 st = float2(uv.x, local);
                float2 dx = ddx(card), dy = ddy(card);
                half4 c = cheer ? SAMPLE_TEXTURE2D_GRAD(_CheerMap, sampler_CheerMap, st, dx, dy)
                                : SAMPLE_TEXTURE2D_GRAD(_BaseMap, sampler_BaseMap, st, dx, dy);

                clip(c.a - 0.5);                                         // empty space: the row behind shows through
                flash = (Hash(float2(seat, row + floor(t * 9.0) * 0.37)) < _PoCrowdFlash * 0.02 && local > 0.55) ? 1.0 : 0.0;

                s = (SurfaceData)0;
                s.albedo = c.rgb * _BaseColor.rgb;
                s.alpha = 1;
                s.specular = half3(0.02, 0.02, 0.02);
                s.smoothness = 0.1;
                s.normalTS = half3(0, 0, 1);
                s.occlusion = 1;
                s.emission = flash * half3(6, 6, 6);
            }

            void CrowdFragment(Varyings input, out half4 outColor : SV_Target0)
            {
                UNITY_SETUP_INSTANCE_ID(input);
                SurfaceData surfaceData;
                float flash;
                CrowdSurface(input, surfaceData, flash);
                InputData inputData;
                InitializeInputData(input, surfaceData.normalTS, inputData);
                InitializeBakedGIData(input, inputData);
                half4 color = UniversalFragmentBlinnPhong(inputData, surfaceData);
                color.rgb = MixFog(color.rgb, inputData.fogCoord);
                outColor = half4(color.rgb, 1);
            }
            ENDHLSL
        }

        // no DepthOnly / DepthNormals: they would write the whole card (unclipped); the crowd stays out of the depth
        // texture (SSAO), depth priming is off in both renderers
        UsePass "Universal Render Pipeline/Simple Lit/META"
    }
    Fallback "Universal Render Pipeline/Simple Lit"
}
