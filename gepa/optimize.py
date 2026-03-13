"""GEPA prompt optimization script.

Optimizes a system prompt (or any text artifact) using GEPA's evolutionary
Pareto search with LLM reflection. The agent modifies this file to change:
  - TASK_LM: the model being optimized
  - REFLECTION_LM: the model doing reflection (should be stronger)
  - SEED: the initial prompt to optimize
  - METRIC: the evaluation function
  - TRAINSET / VALSET: the evaluation data
  - MAX_METRIC_CALLS: budget (number of evaluations)

Usage:
    export OPENAI_API_KEY=...  # or ANTHROPIC_API_KEY
    python optimize.py > run.log 2>&1

Grep-parsable output:
    val_score: 0.85
    best_prompt: <the optimized prompt>
"""

import json
import logging
import gepa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION (agent modifies these)
# ============================================================

# Models
TASK_LM = "openai/gpt-5-mini"          # model being prompted (cheap, fast)
REFLECTION_LM = "openai/gpt-5.4"      # flagship model for better reflection

# Budget
MAX_METRIC_CALLS = 100  # serious run

# Seed prompt to optimize
SEED = {
    "system_prompt": (
        "You are evaluating the quality of a code review comment. "
        "Read the code review below and classify it as either 'good' or 'bad'. "
        "A good review is helpful and specific. A bad review is vague or unhelpful. "
        "Respond with exactly one word: good or bad"
    )
}

# ============================================================
# DATA (agent modifies these)
# ============================================================

def _d(input, answer):
    """Create a data instance with required fields."""
    return {"input": input, "answer": answer, "additional_context": {}}

# --- TRAINSET: 70 labeled code review examples (35 good, 35 bad) ---
TRAINSET = [
    # ---- GOOD reviews (35) ----
    # Bug identification - null/undefined checks
    _d(
        "In `getUserProfile`, line 42: you're accessing `user.address.zipCode` without "
        "checking if `address` is null. If the user never set an address, this will throw "
        "a TypeError in production. Add an optional chain: `user.address?.zipCode`.",
        "good"
    ),
    _d(
        "The `fetchOrders` function on line 87 returns `response.data.orders` but the API "
        "can return `{data: null}` when the user has no orders. This will crash the dashboard. "
        "Suggest: `return response.data?.orders ?? []`.",
        "good"
    ),
    # Bug identification - off-by-one
    _d(
        "Line 31: `for i in range(1, len(items))` skips the first item (index 0). The total "
        "will always be missing one element. Should be `range(len(items))` or `range(0, len(items))`.",
        "good"
    ),
    _d(
        "In the pagination logic, `page_count = total_items / page_size` should use ceiling "
        "division. With 101 items and page_size=10 you get 10.1, which truncates to 10, losing "
        "the last item. Use `math.ceil(total_items / page_size)`.",
        "good"
    ),
    # Bug identification - race conditions
    _d(
        "The `incrementCounter` method reads, increments, and writes back without any locking. "
        "Two threads hitting this concurrently will lose an increment. Either use an atomic "
        "operation like `INCR` in Redis or wrap this in a mutex.",
        "good"
    ),
    _d(
        "In `transferFunds`, you debit account A then credit account B in separate DB calls "
        "with no transaction. If the service crashes between the two, money disappears. Wrap "
        "both operations in a single database transaction.",
        "good"
    ),
    # Bug identification - SQL injection
    _d(
        "Line 55: `query = f\"SELECT * FROM users WHERE name = '{name}'\"` is a textbook SQL "
        "injection vulnerability. If `name` is `'; DROP TABLE users; --` the table is gone. "
        "Use parameterized queries: `cursor.execute('SELECT * FROM users WHERE name = %s', (name,))`.",
        "good"
    ),
    _d(
        "The search endpoint builds the WHERE clause by string concatenation with user input. "
        "This is injectable. Use your ORM's query builder or parameterized statements instead "
        "of raw string formatting.",
        "good"
    ),
    # Bug identification - memory leaks
    _d(
        "The `setInterval` in `usePolling` is never cleared when the component unmounts. Each "
        "mount adds another interval, and after navigating back and forth a few times you'll "
        "have dozens of timers running. Return a cleanup function from `useEffect`.",
        "good"
    ),
    _d(
        "In the event listener setup on line 23, you attach a new `resize` handler every time "
        "`render()` is called but never remove the old one. Over time this leaks memory and "
        "causes duplicate work. Move the listener to `componentDidMount` with cleanup in `componentWillUnmount`.",
        "good"
    ),
    # Concrete fix suggestions with code
    _d(
        "The retry logic in `callExternalAPI` retries immediately on failure, which will "
        "hammer the downstream service. Add exponential backoff: "
        "`await sleep(Math.pow(2, attempt) * 1000)` between retries.",
        "good"
    ),
    _d(
        "In `parseCSV`, you're splitting on comma but not handling quoted fields. A field like "
        "`\"Smith, John\"` will be incorrectly split into two columns. Use the `csv` module "
        "instead of `line.split(',')`.",
        "good"
    ),
    # Edge cases
    _d(
        "What happens in `calculateDiscount` when `quantity` is 0? Line 18 divides "
        "`totalPrice / quantity` which will throw a ZeroDivisionError. Add a guard: "
        "`if quantity == 0: return 0`.",
        "good"
    ),
    _d(
        "The `formatPhoneNumber` function assumes US format (10 digits) but doesn't validate "
        "input length. Passing an international number like +44 will silently produce garbage "
        "output. Add length validation and consider supporting E.164 format.",
        "good"
    ),
    _d(
        "The date comparison `if date_str > '2024-01-01'` works only because ISO format sorts "
        "lexicographically. But if someone changes the date format to DD/MM/YYYY this silently "
        "breaks. Parse both to datetime objects for a robust comparison.",
        "good"
    ),
    # Performance issues with complexity
    _d(
        "The nested loop on lines 12-18 iterates over `users` x `permissions`, making this "
        "O(n*m). With 10k users and 500 permissions, that's 5M iterations per request. Build "
        "a set from `permissions` first for O(1) lookups, making it O(n+m).",
        "good"
    ),
    _d(
        "Calling `JSON.parse(JSON.stringify(obj))` for deep clone inside a loop that runs "
        "10,000 times is extremely slow. Use `structuredClone(obj)` or a library like "
        "lodash's `cloneDeep` for better performance.",
        "good"
    ),
    _d(
        "The `getRecommendations` query does a full table scan on `orders` (2M rows) because "
        "there's no index on `customer_id`. Add `CREATE INDEX idx_orders_customer ON orders(customer_id)` "
        "to bring this from ~3s to <10ms.",
        "good"
    ),
    # Security issues
    _d(
        "The JWT token is stored in `localStorage`, which is accessible to any XSS attack on "
        "the page. Consider using httpOnly cookies instead, which JavaScript cannot read. This "
        "is especially important since you're rendering user-generated HTML.",
        "good"
    ),
    _d(
        "Line 34: `eval(userInput)` in the template engine gives users arbitrary code execution. "
        "Even if this is an internal tool, an attacker who compromises any user account gets "
        "full server access. Use a sandboxed template library instead.",
        "good"
    ),
    # Logic errors
    _d(
        "In `isEligibleForDiscount`, the condition `age > 65 || memberSince < 2020` should "
        "use AND, not OR. Right now every member who joined before 2020 gets a senior discount "
        "regardless of age. Based on the spec, both conditions must be true.",
        "good"
    ),
    _d(
        "The `validatePassword` function checks `password.length > 8` but the requirements "
        "doc says 'at least 8 characters', which means `>=`. A password of exactly 8 chars "
        "will be incorrectly rejected.",
        "good"
    ),
    # Error handling
    _d(
        "The catch block on line 45 is `except Exception: pass`. This silently swallows all "
        "errors including things like `KeyboardInterrupt` and `SystemExit`. At minimum, log "
        "the exception. Better: catch only the specific exception you expect (e.g., `ValueError`).",
        "good"
    ),
    _d(
        "In `processPayment`, if the Stripe API returns a 500, you return `None` with no "
        "logging or error propagation. The caller then tries to access `.transaction_id` on "
        "`None` and crashes with an unhelpful error. Raise a custom `PaymentError` with context.",
        "good"
    ),
    # API design
    _d(
        "The `deleteUser` endpoint returns 200 with an empty body on success, but also returns "
        "200 when the user doesn't exist. The client can't distinguish success from no-op. "
        "Return 204 for success and 404 when the user doesn't exist.",
        "good"
    ),
    # Type safety
    _d(
        "The `total` prop is typed as `number` but `formatCurrency(total)` crashes when "
        "`total` is undefined because the parent component doesn't always pass it. Either "
        "add a default value (`total = 0`) or make the prop required in the parent.",
        "good"
    ),
    # Testing gaps
    _d(
        "The test for `parseEmail` only checks valid emails. Add cases for empty string, "
        "missing @, double @, unicode domains, and very long local parts (>64 chars). The "
        "RFC 5321 edge cases are where parsers usually break.",
        "good"
    ),
    # Resource management
    _d(
        "The file handle opened on line 12 with `open('data.csv')` is never closed if an "
        "exception occurs on line 15. Use a `with` statement: `with open('data.csv') as f:` "
        "to ensure the file is always properly closed.",
        "good"
    ),
    # Concurrency
    _d(
        "The `cache` dict is shared across threads but has no synchronization. Concurrent "
        "reads and writes to a plain dict can cause `RuntimeError: dictionary changed size "
        "during iteration`. Use `threading.Lock` or `concurrent.futures` with a thread-safe cache.",
        "good"
    ),
    # Data integrity
    _d(
        "In `updateInventory`, you set `quantity = quantity - ordered` but don't check if "
        "`quantity >= ordered` first. This can make inventory go negative, causing downstream "
        "issues with fulfillment. Add a check and raise `InsufficientStockError` if needed.",
        "good"
    ),
    # Encoding issues
    _d(
        "The `readFile` function uses `open(path, 'r')` which defaults to the system encoding "
        "(often latin-1 on Windows). If anyone uploads a UTF-8 file with emoji or CJK characters, "
        "it will crash. Use `open(path, 'r', encoding='utf-8')` explicitly.",
        "good"
    ),
    # Configuration
    _d(
        "The API timeout is hardcoded to 30 seconds on line 5. In production, the downstream "
        "service sometimes takes 45s during peak load, causing spurious failures. Make this "
        "configurable via environment variable with 30s as default.",
        "good"
    ),
    # Logging
    _d(
        "Line 67: `log.info(f'Processing payment for user {user.email}, card {card_number}')` "
        "is logging full credit card numbers to the application log. This violates PCI-DSS. "
        "Mask the card number: `card_number[-4:].rjust(len(card_number), '*')`.",
        "good"
    ),
    # Deprecation
    _d(
        "You're using `datetime.utcnow()` which is deprecated in Python 3.12 and returns a "
        "naive datetime. Use `datetime.now(datetime.timezone.utc)` instead, which returns an "
        "aware datetime and is the recommended approach going forward.",
        "good"
    ),
    # Correctness
    _d(
        "The `average` function returns `sum(values) / len(values)` but doesn't handle the "
        "empty list case. With `values = []`, this raises ZeroDivisionError. Also consider "
        "whether you want integer or float division -- `sum([1,2]) / 2` gives 1.5 in Python 3 "
        "but 1 in Python 2.",
        "good"
    ),

    # ---- BAD reviews (35) ----
    # Rubber stamps
    _d("LGTM", "bad"),
    _d("Looks good to me, ship it!", "bad"),
    _d("LGTM! Nice work.", "bad"),
    _d("Approved. No comments.", "bad"),
    _d("+1, let's merge this.", "bad"),
    _d("Seems fine. Approving.", "bad"),
    # Vague suggestions
    _d("This could be better.", "bad"),
    _d("Consider refactoring this part.", "bad"),
    _d("I think there might be a cleaner way to do this.", "bad"),
    _d("Maybe add some error handling?", "bad"),
    _d("This function is too long, can you break it up?", "bad"),
    _d("The naming could be improved here.", "bad"),
    _d("Have you considered edge cases?", "bad"),
    _d("This doesn't feel right, but I can't put my finger on it.", "bad"),
    # Wrong or misleading
    _d(
        "You should use `var` instead of `let` here for better hoisting behavior. "
        "`let` can cause issues with closures in loops.",
        "bad"
    ),
    _d(
        "This would be faster if you used a linked list instead of an array for random access.",
        "bad"
    ),
    _d(
        "You don't need to close database connections in Python, the garbage collector handles it.",
        "bad"
    ),
    # Repeating what the code does
    _d(
        "I see this function takes a list of users and iterates over them, checking if each "
        "one is active, and then returns the filtered list.",
        "bad"
    ),
    _d(
        "This endpoint accepts a POST request with a JSON body containing the user's name "
        "and email, then saves it to the database and returns a 201 status.",
        "bad"
    ),
    _d(
        "So this loops through the items, adds each price to a running total, and then "
        "returns the total at the end.",
        "bad"
    ),
    # Style nitpicks without substance
    _d("Nit: can you add a blank line between these two functions?", "bad"),
    _d("Nit: prefer single quotes over double quotes for strings.", "bad"),
    _d("Can you rename `x` to something more descriptive? Maybe `value`?", "bad"),
    _d("I'd move this import to the top of the file for consistency.", "bad"),
    _d(
        "Minor: the indentation on line 45 uses a tab instead of spaces. "
        "Can you fix that?",
        "bad"
    ),
    # Overly long but says nothing
    _d(
        "I took a look at this PR and overall it seems like a reasonable approach to the "
        "problem. I think the general structure makes sense and the code is readable. There "
        "are a few things I might do differently but nothing blocking. Nice job overall.",
        "bad"
    ),
    _d(
        "I've been thinking about this for a while and I think it's generally fine. The "
        "approach you've taken is one way to do it, and I think it will work. I don't have "
        "any major concerns. Maybe in the future we could revisit this area of the code but "
        "for now this is acceptable.",
        "bad"
    ),
    _d(
        "Thanks for working on this! I know this area of the codebase is tricky. I looked "
        "through the changes and everything seems reasonable. Let me know if you have any "
        "questions about anything.",
        "bad"
    ),
    # Asking questions that should be obvious from context
    _d("What does this function do?", "bad"),
    _d("Why did you choose this approach?", "bad"),
    _d("Is this a new feature or a bug fix?", "bad"),
    # Bike-shedding
    _d(
        "I think we should use camelCase instead of snake_case for this variable, even though "
        "the rest of the codebase uses snake_case. camelCase is more common in JavaScript.",
        "bad"
    ),
    _d(
        "Can we debate whether this should be a class method or a standalone function? I have "
        "strong feelings about OOP patterns here.",
        "bad"
    ),
    # Lazy approval with false authority
    _d("I didn't read the whole thing but it looks fine from the diff summary.", "bad"),
    _d(
        "I trust you on this one. Approving without a thorough review since we need to "
        "ship by Friday.",
        "bad"
    ),
]

# --- VALSET: 30 labeled code review examples (15 good, 15 bad) ---
VALSET = [
    # ---- GOOD reviews (15) ----
    _d(
        "In `sendEmail`, the `to` parameter is passed directly to the SMTP library without "
        "any validation. An attacker could inject headers with `\\r\\nBCC: spam@evil.com` and "
        "use your server as an open relay. Sanitize the input or use a library that handles this.",
        "good"
    ),
    _d(
        "Line 22: `setTimeout(callback, delay)` where `delay` comes from user input. A "
        "negative value or extremely large number will cause unexpected behavior. Clamp it: "
        "`Math.max(0, Math.min(delay, 300000))`.",
        "good"
    ),
    _d(
        "The `mergeConfigs` function uses the spread operator `{...defaults, ...userConfig}` "
        "but this is a shallow merge. Nested objects like `userConfig.database.pool` will "
        "completely overwrite `defaults.database`, losing `defaults.database.host`. Use a "
        "deep merge utility.",
        "good"
    ),
    _d(
        "The `binarySearch` implementation returns -1 when the element isn't found, but the "
        "caller on line 93 checks `if (result)` which treats 0 (a valid index) as not found. "
        "Change the check to `if (result !== -1)` or `if (result >= 0)`.",
        "good"
    ),
    _d(
        "In the `handleUpload` endpoint, `req.file.size` is checked after the entire file is "
        "already in memory. A 2GB upload will OOM the server before validation runs. Set "
        "`limits: { fileSize: 10 * 1024 * 1024 }` in the multer config to reject early.",
        "good"
    ),
    _d(
        "The `Promise.all` on line 56 will reject if any single notification fails, causing "
        "the remaining notifications to be lost. Use `Promise.allSettled` instead so failures "
        "are logged but don't block successful sends.",
        "good"
    ),
    _d(
        "The regex `^\\d{3}-\\d{4}$` for phone validation is too restrictive -- it only "
        "matches 7-digit numbers. US phone numbers need `^\\d{3}-\\d{3}-\\d{4}$` (10 digits), "
        "and ideally you'd support international formats too.",
        "good"
    ),
    _d(
        "The `comparePasswords` function uses `===` for string comparison, which is vulnerable "
        "to timing attacks. Use `crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b))` to "
        "prevent an attacker from guessing the hash character by character.",
        "good"
    ),
    _d(
        "In `generateReport`, the SQL query uses `ORDER BY date DESC LIMIT 1000` without any "
        "date range filter. As the table grows, this will scan increasingly more rows. Add a "
        "WHERE clause on `date >= NOW() - INTERVAL 90 DAY` and add an index on `date`.",
        "good"
    ),
    _d(
        "The `convertCurrency` function caches exchange rates forever after first fetch. "
        "Rates change constantly -- a stale rate could cause users to be charged the wrong "
        "amount. Add a TTL of 15 minutes and re-fetch when expired.",
        "good"
    ),
    _d(
        "Line 8: `del items[i]` inside a `for i in range(len(items))` loop modifies the list "
        "while iterating, which will skip elements and eventually raise IndexError. Build a "
        "new list with a comprehension: `items = [x for x in items if not should_remove(x)]`.",
        "good"
    ),
    _d(
        "The `hashPassword` function uses MD5, which is cryptographically broken and can be "
        "brute-forced in seconds with modern GPUs. Use bcrypt or argon2 with a proper salt "
        "and work factor of at least 10.",
        "good"
    ),
    _d(
        "In the WebSocket handler, if `socket.send()` throws because the client disconnected, "
        "the error propagates up and crashes the entire server process. Wrap the send in a "
        "try/catch and remove dead connections from the client list.",
        "good"
    ),
    _d(
        "The `calculateShipping` function returns a float like 9.999999999 due to floating-point "
        "arithmetic. For money calculations, use `Decimal` (Python) or integer cents to avoid "
        "rounding errors that compound across thousands of orders.",
        "good"
    ),
    _d(
        "The `env.DATABASE_URL` is used directly in the connection string without URL-encoding. "
        "If the password contains special characters like `@` or `/`, the connection will fail "
        "silently or connect to the wrong host. Use `urllib.parse.quote_plus` on the password.",
        "good"
    ),

    # ---- BAD reviews (15) ----
    _d("Looks great, no issues!", "bad"),
    _d("Ship it!", "bad"),
    _d("Maybe try a different approach here.", "bad"),
    _d("This is a bit complex. Can you simplify it?", "bad"),
    _d(
        "I noticed this function calls another function which then calls a third function "
        "that processes the data and returns it back up the chain.",
        "bad"
    ),
    _d("Nit: extra whitespace on line 33.", "bad"),
    _d("Can you add a comment explaining what this does?", "bad"),
    _d(
        "Overall this is a solid PR. Well done! I appreciate the effort you put into this. "
        "The code is clean and readable. I have no concerns.",
        "bad"
    ),
    _d("Why not use a ternary operator here instead of if/else?", "bad"),
    _d(
        "I'd suggest using TypeScript for this file, it would catch a lot of bugs.",
        "bad"
    ),
    _d("Did you run the tests?", "bad"),
    _d(
        "This is similar to something I wrote last quarter, might be worth looking at that "
        "as a reference.",
        "bad"
    ),
    _d("Could you add more tests?", "bad"),
    _d(
        "I glanced at this quickly between meetings. Seems okay, approving so it doesn't "
        "block the release.",
        "bad"
    ),
    _d(
        "You might want to use `const` instead of `let` on line 12, even though the value "
        "is reassigned later in the function.",
        "bad"
    ),
]

# ============================================================
# RUN
# ============================================================

def main():
    log.info("Starting GEPA optimization")
    log.info(f"Task LM: {TASK_LM}")
    log.info(f"Reflection LM: {REFLECTION_LM}")
    log.info(f"Budget: {MAX_METRIC_CALLS} metric calls")
    log.info(f"Train: {len(TRAINSET)} examples, Val: {len(VALSET)} examples")
    log.info(f"Seed prompt: {SEED['system_prompt'][:100]}...")

    result = gepa.optimize(
        seed_candidate=SEED,
        trainset=TRAINSET,
        valset=VALSET,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        display_progress_bar=True,
    )

    # Extract results
    best = result.best_candidate
    best_idx = result.best_idx
    val_score = result.val_aggregate_scores[best_idx]

    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)

    # Grep-parsable output
    print(f"\nval_score: {val_score:.6f}")
    print(f"best_prompt: {json.dumps(best)}")

    # Human-readable
    log.info(f"Val score: {val_score}")
    log.info(f"Best prompt: {best.get('system_prompt', str(best))}")
    log.info(f"Candidates explored: {len(result.candidates)}")
    log.info(f"Total metric calls: {result.total_metric_calls}")
    log.info("Optimization completed.")


if __name__ == "__main__":
    main()
