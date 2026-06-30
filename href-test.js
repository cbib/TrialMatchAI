// Test the href encoding behavior

const pt = { id: null };
const href1 = "#/p/" + encodeURIComponent(String(pt.id));
console.log("href with null id:", href1);

const pt2 = { id: undefined };
const href2 = "#/p/" + encodeURIComponent(String(pt2.id));
console.log("href with undefined id:", href2);

const pt3 = { id: "patient123" };
const href3 = "#/p/" + encodeURIComponent(String(pt3.id));
console.log("href with string id:", href3);

// Now test the route matching
const location_hash1 = "#/p/null";
const m1 = location_hash1.match(/^#\/p\/(.+)$/);
console.log("Route match for #/p/null:", m1 ? m1[1] : "no match");

const location_hash2 = "#/p/patient123";
const m2 = location_hash2.match(/^#\/p\/(.+)$/);
console.log("Route match for #/p/patient123:", m2 ? m2[1] : "no match");

// Simulate finding patient with null id
const PATIENTS = [
  { patient: { id: null }, trials: [] },
  { patient: { id: "p1" }, trials: [] }
];

const id = "null";
const p = PATIENTS.find(x => String((x.patient || {}).id) === id);
console.log("Found patient with id 'null':", p);

const id2 = "p1";
const p2 = PATIENTS.find(x => String((x.patient || {}).id) === id2);
console.log("Found patient with id 'p1':", p2);
