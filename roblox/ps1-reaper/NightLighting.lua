-- Night graveyard look for the PS1 reaper. Put this in a Script inside ServerScriptService.
local Lighting = game:GetService("Lighting")

-- Roblox ignores the classic fog below while an Atmosphere exists, and hard distance fog is the PS1 look
local atmosphere = Lighting:FindFirstChildOfClass("Atmosphere")
if atmosphere then
	atmosphere:Destroy()
end

Lighting.ClockTime = 0
Lighting.Brightness = 2
Lighting.Ambient = Color3.fromRGB(35, 40, 70)
Lighting.OutdoorAmbient = Color3.fromRGB(70, 80, 130)
Lighting.FogColor = Color3.fromRGB(18, 22, 48)
Lighting.FogStart = 15
Lighting.FogEnd = 140

local sky = Lighting:FindFirstChildOfClass("Sky") or Instance.new("Sky")
sky.MoonAngularSize = 35 -- big full moon like the reference
sky.StarCount = 3000
sky.Parent = Lighting

local grade = Lighting:FindFirstChild("PS1Grade") or Instance.new("ColorCorrectionEffect")
grade.Name = "PS1Grade"
grade.Contrast = 0.15
grade.Saturation = 0.1
grade.TintColor = Color3.fromRGB(205, 210, 255)
grade.Parent = Lighting
