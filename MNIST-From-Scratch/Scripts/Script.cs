namespace MNIST_From_Scratch.Scripts
{
    public abstract class Script
    {
        public abstract string Name { get; }
        public abstract void Run(Dictionary<string, string> arguments);

        protected T GetArgument<T>(Dictionary<string, string> arguments, string key)
        {
            if (arguments.TryGetValue(key, out var value))
            {
                return (T)Convert.ChangeType(value, typeof(T));
            }
            throw new ArgumentException($"Argument '{key}' is required.");
        }

        protected bool TryGetArgument<T>(Dictionary<string, string> arguments, string key, out T? value)
        {
            bool success = arguments.TryGetValue(key, out var v);
            value = success ? (T?)Convert.ChangeType(v, typeof(T)) : default;
            return success;
        }
    }
}
