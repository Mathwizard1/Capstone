# Graph Matching Approaches

## 1. Batch-based Graph Matching [Time-based]

**Entities:** Customers (orders), Restaurants, Delivery agents  
**Tuple:** `(customer, restaurant, driver)`

**Description:** Collect orders for Δ time → then match together instead of matching immediately

**Benefits:**
- Route optimization: Combine multiple orders into one trip
- Better driver utilization: Avoid sending driver for single order
- Higher profit / efficiency

**Datasets:**
- https://www.kaggle.com/datasets/jayjoshi37/daily-food-delivery-orders-and-delivery-time
- https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset - na

---

## 2. Spatial-Compatibility Graph [Constraint-based]

**Tuple:** `(customer, restaurant, driver)`

**Description:** Build edges only if delivery is feasible

**Edge Rules:**
- `customer → restaurant`: Order exists
- `restaurant → driver`: `distance(driver, restaurant) ≤ d`
- `customer → driver`: `estimated delivery time ≤ τ`

**Use Case:** Food delivery apps (Swiggy, Uber Eats)

**Datasets:** Same as above

---

## 3. Priority/Urgency Graph [Healthcare Matching] [Priority-based]

**Tuple:** `(patient, hospital, resource)`

**Description:** Decisions are immediate. Patient arrives → match instantly based on urgency level, hospital feasibility, and resource availability

**Benefits:**
- High urgency patients prioritized
- Resource constraints satisfied
- Response time minimized

**Edge Rules:**
- `patient → hospital`: Hospital can treat patient condition
- `hospital → resource`: Resource available (bed/doctor)
- `patient → resource`: Urgency + compatibility

**Note:** Becomes k-partite with: `(patient, hospital, resource, time, severity, location)`

**Datasets:**
- https://www.kaggle.com/datasets/montassarba/mimic-iv-clinical-database-demo-2-2
- https://www.kaggle.com/datasets/asjad99/mimiciii

---

## 4. Scheduling Graph [Jobs → Servers → Time] [Scheduling-based]

**Tuple:** `(job, server, time)`

**Description:** Jobs arrive online → assign to server along with a time slot. No batching, but scheduling is important.

**Decision Factors:**
- Which server
- At what time

**Edge Rules:**
- `job → server`: Server can process the job
- `server → time`: Server available at that time slot
- `job → time`: Deadline / release time constraint

**Goals:**
- Deadlines satisfied
- Server capacity respected
- Utilization maximized

**Why This Works:**
- No delay: decisions made on arrival
- Time-aware: scheduling improves efficiency
- Resource utilization: better server usage

**Note:** Becomes k-partite with: `(job, server, time, priority, resource_type)`

**Dataset:** https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample
