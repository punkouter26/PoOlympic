using System.IO;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>Static rule for the style sheets (UI/Tokens.uss): every colour and every font size is a --po-* token,
    /// so one edit of Tokens.uss restyles the menu and the HUD together. ScreenBoard.uss keeps its own font sizes: it
    /// is the stadium screen, a render texture at another scale.</summary>
    public class UiTokenTests
    {
        [Test]
        public void ColoursAndFontSizesAreTokens()
        {
            var raw = new Regex(@"(?<prop>font-size|[\w-]*color)\s*:\s*(?<value>[^;]*?(rgba?\(|#[0-9a-fA-F]{3,8}\b|\d+px)[^;]*);");
            foreach (var f in Directory.GetFiles(Path.Combine(Application.dataPath, "PoOlympic", "UI"), "*.uss"))
            {
                string name = Path.GetFileName(f);
                if (name == "Tokens.uss") continue;
                foreach (Match m in raw.Matches(File.ReadAllText(f)))
                {
                    if (name == "ScreenBoard.uss" && m.Groups["prop"].Value == "font-size") continue;
                    Assert.Fail($"{name}: '{m.Value}' — use a --po-* token from Tokens.uss");
                }
            }
        }
    }
}
