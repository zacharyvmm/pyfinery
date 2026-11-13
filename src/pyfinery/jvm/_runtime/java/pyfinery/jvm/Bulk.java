package pyfinery.jvm;

import tools.refinery.store.model.Interpretation;
import tools.refinery.store.query.resultset.ResultSet;
import tools.refinery.store.tuple.Tuple;

/**
 * Moves many tuples across the Python-Java boundary in one call: a put called from Python costs microseconds, one
 * inside this loop a fraction of that.
 */
public final class Bulk {
	private Bulk() {
		throw new IllegalStateException("This is a static utility class and should not be instantiated directly");
	}

	/**
	 * Puts {@code value} on each tuple of {@code ids}, which holds the tuples one after another, {@code arity} ids
	 * each. Putting the interpretation's default value removes the tuples.
	 */
	public static <T> void put(Interpretation<T> interpretation, int[] ids, int arity, T value) {
		if (arity < 0 || (arity == 0 ? ids.length != 0 : ids.length % arity != 0)) {
			throw new IllegalArgumentException("ids holds " + ids.length + " ids, not tuples of " + arity);
		}
		if (arity == 0) {
			interpretation.put(Tuple.of(), value);
			return;
		}
		for (int i = 0; i < ids.length; i += arity) {
			interpretation.put(tuple(ids, i, arity), value);
		}
	}

	/** The keys of {@code resultSet}, one after another, {@code arity} ids each. */
	public static int[] keys(ResultSet<?> resultSet, int arity) {
		var keys = new int[resultSet.size() * arity];
		var cursor = resultSet.getAll();
		int i = 0;
		while (cursor.move()) {
			var key = cursor.getKey();
			for (int j = 0; j < arity; j++) {
				keys[i++] = key.get(j);
			}
		}
		return keys;
	}

	private static Tuple tuple(int[] ids, int start, int arity) {
		return switch (arity) {
			case 1 -> Tuple.of(ids[start]);
			case 2 -> Tuple.of(ids[start], ids[start + 1]);
			case 3 -> Tuple.of(ids[start], ids[start + 1], ids[start + 2]);
			case 4 -> Tuple.of(ids[start], ids[start + 1], ids[start + 2], ids[start + 3]);
			default -> Tuple.of(java.util.Arrays.copyOfRange(ids, start, start + arity));
		};
	}
}
