// The shape of a test suite. Each model exports one, and the developer page of the site runs it.

// What a test can send while it runs. say() adds a log line. metric() records one number with a unit.
export type TestContext = {
    say(line: string): void;
    metric(name: string, value: number, unit: string): void;
};

// The end of a test: passed or not, and one line that says why.
export type TestOutcome = { passed: boolean; summary: string };

export type TestCase = {
    // The id in the page address, for example ?test=golden-image.
    id: string;
    title: string;
    // One sentence: what the test checks.
    description: string;
    run(context: TestContext): Promise<TestOutcome>;
};

export type TestSuite = { model: string; title: string; tests: readonly TestCase[] };
