using Godot;
using Godot.Collections;

namespace GhostMerc.LogisticsVNext;

public static class StoryPackageRuntime
{
    private static string NormalizePath(string rawPath, string baseDir = "")
    {
        var path = (rawPath ?? string.Empty).Trim().Replace("\\", "/");
        if (string.IsNullOrEmpty(path))
        {
            return path;
        }
        if (path.StartsWith("res://") || path.StartsWith("user://") || path.Contains(":/") || path.StartsWith("/"))
        {
            return path;
        }
        if (!string.IsNullOrEmpty(baseDir))
        {
            return baseDir.PathJoin(path).SimplifyPath();
        }
        return ProjectSettings.GlobalizePath("res://").PathJoin(path).SimplifyPath();
    }

    private static Variant ReadJson(string path)
    {
        var normalizedPath = NormalizePath(path);
        if (!FileAccess.FileExists(normalizedPath))
        {
            GD.PushError($"StoryPackageRuntime: missing JSON file: {normalizedPath}");
            return Variant.CreateFrom(null);
        }
        using var handle = FileAccess.Open(normalizedPath, FileAccess.ModeFlags.Read);
        if (handle is null)
        {
            GD.PushError($"StoryPackageRuntime: could not open JSON file: {normalizedPath}");
            return Variant.CreateFrom(null);
        }
        var parsed = Json.ParseString(handle.GetAsText());
        return parsed;
    }

    public static Dictionary LoadFromRuntimePointer(string pointerPath = "res://runtime/latest_story_package.json")
    {
        var pointerPayload = ReadJson(pointerPath).AsGodotDictionary();
        if (pointerPayload.Count == 0)
        {
            return new Dictionary();
        }
        var packagePath = pointerPayload.TryGetValue("story_package_path", out var packageVariant)
            ? packageVariant.AsString()
            : string.Empty;
        if (string.IsNullOrEmpty(packagePath))
        {
            GD.PushError("StoryPackageRuntime: runtime pointer does not include story_package_path.");
            return new Dictionary();
        }
        return LoadFromPackagePath(packagePath);
    }

    public static Dictionary LoadFromPackagePath(string packagePath)
    {
        var normalizedPackage = NormalizePath(packagePath);
        var packagePayload = ReadJson(normalizedPackage).AsGodotDictionary();
        if (packagePayload.Count == 0)
        {
            return new Dictionary();
        }
        var packageDir = normalizedPackage.GetBaseDir();
        var sequencePath = packagePayload.TryGetValue("sequence_file", out var sequenceVariant)
            ? NormalizePath(sequenceVariant.AsString(), packageDir)
            : NormalizePath("sequence.json", packageDir);
        var sequencePayload = ReadJson(sequencePath).AsGodotDictionary();
        if (sequencePayload.Count == 0)
        {
            return new Dictionary();
        }
        return new Dictionary
        {
            { "package_path", normalizedPackage },
            { "package_dir", packageDir },
            { "package", packagePayload },
            { "sequence_path", sequencePath },
            { "sequence", sequencePayload },
            { "presentation_modes", packagePayload.TryGetValue("presentation_modes", out var modesVariant) ? modesVariant : new Dictionary() }
        };
    }
}
