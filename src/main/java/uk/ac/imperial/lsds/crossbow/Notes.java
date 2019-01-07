package uk.ac.imperial.lsds.crossbow;

public class Notes {}

/*
 * A gradient can be freed either by:
 * a) a worker thread, after it has been used to calculate momentum, or 
 * b) a result handler, after it has been accumulated.
 * 
 * The logic is as follows:
 * 
 * model.apply (Gradient gradient, hasMomentum) {
 * 		model += gradient;
 * 		if (hasMomentum) {
 * 			g.retain ();
 * 			g.freeAfterMomentumComputation ();
 * 			[ Swap gradients ]
 * 			q = last;
 * 			last = g;
 * 			[ Try to free previous gradient ]
 * 			if (! q.isRetained ())
 * 				error
 * 			free (q); [ Will block until `q` has been accumulated ]
 * 		} else {
 * 			g.freeAfterAccumulation ();
 * 		}
 * }
 * 
 * freeSlot (int ndx) {
 * 		Gradient g = gradients.elementAt (ndx);
 * 		model.accumulate (g);
 * 		g.setAccumulated ();
 * 		if (! g.freeOnceAccumulated ())
 * 			free (g); [ Will not block, since it has been accumulated ]
 * }
 * 
 */
