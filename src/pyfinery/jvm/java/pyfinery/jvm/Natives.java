package pyfinery.jvm;

/**
 * Loads OR-tools' native libraries from a directory they were unpacked into once.
 * <p>
 * OR-tools' own {@code Loader} unpacks them into a new temporary directory on every start, which takes seconds.
 * {@link System#load} binds a library to the class loader of its caller, so it is called here, from a class on the
 * application class path like OR-tools', and not from Python.
 * </p>
 */
public final class Natives {
	private Natives() {
		throw new IllegalStateException("This is a static utility class and should not be instantiated directly");
	}

	/**
	 * Loads OR-tools' native libraries at {@code paths}, in that order (on Windows, the libraries jniortools needs
	 * come first, as OR-tools' own {@code Loader} loads them), and tells its {@code Loader} not to unpack them again.
	 */
	public static void loadOrTools(String[] paths) throws ReflectiveOperationException {
		for (var path : paths) {
			System.load(path);
		}
		var loaded = Class.forName("com.google.ortools.Loader").getDeclaredField("loaded");
		loaded.setAccessible(true);
		loaded.setBoolean(null, true);
	}
}
