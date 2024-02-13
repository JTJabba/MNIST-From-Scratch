using MNIST_From_Scratch.Scripts;

namespace MNIST_From_Scratch
{
    class Program
    {
        static List<Script> scripts = new List<Script>()
        {
            new DecodeImageScript(),
            new GradientCheckScript(),
            new InferenceScript(),
            new TrainingScript(),
        };

        static void Main(string[] args)
        {
            while (args.Length == 0 || args[0].Length == 0 | args[0] == "--help")
            {
                Printhelp();
                args = Console.ReadLine()?.Split(' ') ?? new string[0];
            }

            var scriptName = args[0];
            var script = scripts.FirstOrDefault(s => s.Name.Equals(scriptName, StringComparison.OrdinalIgnoreCase));

            if (script == null)
            {
                Console.WriteLine($"Script '{scriptName}' not found.");
                return;
            }

            var arguments = ParseArguments(args.Skip(1).ToArray());
            try
            {
                script.Run(arguments);
            } catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        static Dictionary<string, string> ParseArguments(string[] args)
        {
            var arguments = new Dictionary<string, string>();
            foreach (var arg in args)
            {
                var parts = arg.Split('=');
                if (parts.Length == 2)
                {
                    arguments[parts[0]] = parts[1];
                }
            }
            return arguments;
        }

        static void Printhelp()
        {
            Console.WriteLine("Available scripts:");
            foreach (var s in scripts)
            {
                Console.WriteLine($"- {s.Name}");
            }
        }
    }
}
