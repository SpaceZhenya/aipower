/**
 * browser-agent (Qwen edition)
 *
 * An AI agent that controls a real Chromium browser to accomplish a
 * natural-language goal, using Qwen (via Alibaba Cloud DashScope's
 * OpenAI-compatible API) for decision-making and Playwright for browser
 * control.
 *
 * Usage (local):
 *   npm install
 *   npm run install-browsers
 *   export DASHSCOPE_API_KEY=sk-...
 *   node agent.js "search for the weather in Tokyo and tell me today's forecast"
 *
 * Config (env vars):
 *   DASHSCOPE_API_KEY   - required, your DashScope API key
 *   DASHSCOPE_BASE_URL  - optional, defaults to the international endpoint.
 *                         Use https://dashscope.aliyuncs.com/compatible-mode/v1
 *                         for the mainland China endpoint instead.
 *   QWEN_MODEL          - optional, defaults to "qwen-plus"
 *                         (other options: qwen-max, qwen-turbo, etc.)
 *   HEADLESS            - optional, "true"/"false", defaults to true
 *
 * Usage (GitHub Actions):
 *   See .github/workflows/run-agent.yml — runs headless, goal passed as
 *   a workflow_dispatch input, API key pulled from repo secrets.
 */

import OpenAI from "openai";
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "fs";

const client = new OpenAI({
  apiKey: process.env.DASHSCOPE_API_KEY,
  baseURL:
    process.env.DASHSCOPE_BASE_URL ||
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
});

const MODEL = process.env.QWEN_MODEL || "qwen-plus";
const MAX_STEPS = 20; // safety cap so the agent can't loop forever

// ---------------------------------------------------------------------------
// 1. Define the tools Qwen is allowed to use to control the browser.
//    (OpenAI-style function-calling schema, which DashScope's compatible
//    endpoint expects.)
// ---------------------------------------------------------------------------
const tools = [
  {
    type: "function",
    function: {
      name: "navigate",
      description: "Go to a URL in the browser.",
      parameters: {
        type: "object",
        properties: { url: { type: "string" } },
        required: ["url"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "click",
      description:
        "Click an element on the page, identified by a CSS selector or visible text.",
      parameters: {
        type: "object",
        properties: {
          selector: {
            type: "string",
            description: "CSS selector, or exact visible text of the element.",
          },
        },
        required: ["selector"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "type_text",
      description: "Type text into an input field identified by a CSS selector.",
      parameters: {
        type: "object",
        properties: {
          selector: { type: "string" },
          text: { type: "string" },
          press_enter: {
            type: "boolean",
            description: "Whether to press Enter after typing.",
          },
        },
        required: ["selector", "text"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "read_page",
      description:
        "Get a simplified text snapshot of the current page's visible content (for deciding what to do next).",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "finish",
      description:
        "Call this when the task is complete, with the final answer or result for the user.",
      parameters: {
        type: "object",
        properties: { result: { type: "string" } },
        required: ["result"],
      },
    },
  },
];
// Note: unlike a vision-model setup, this skips a "screenshot" tool.
// Standard Qwen tool-calling models don't reliably accept images injected
// as a tool result, so page understanding here relies on read_page's text
// snapshot. If you switch to a Qwen-VL (vision) model, you can wire
// screenshots back in via an image content block in the follow-up message.

// ---------------------------------------------------------------------------
// 2. Execute a tool call against the live Playwright page.
// ---------------------------------------------------------------------------
async function executeTool(page, name, input) {
  switch (name) {
    case "navigate":
      await page.goto(input.url, { waitUntil: "domcontentloaded" });
      return { ok: true, url: page.url() };

    case "click": {
      const locator = looksLikeSelector(input.selector)
        ? page.locator(input.selector).first()
        : page.getByText(input.selector, { exact: false }).first();
      await locator.click({ timeout: 5000 });
      return { ok: true };
    }

    case "type_text": {
      const locator = looksLikeSelector(input.selector)
        ? page.locator(input.selector).first()
        : page.getByPlaceholder(input.selector).first();
      await locator.fill(input.text, { timeout: 5000 });
      if (input.press_enter) await locator.press("Enter");
      return { ok: true };
    }

    case "read_page": {
      const text = await page.evaluate(() => document.body.innerText);
      return { text: text.slice(0, 4000) }; // keep it bounded
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

function looksLikeSelector(s) {
  return /^[.#\[]|^[a-z]+\[|^(div|span|a|button|input|form)\b/i.test(s);
}

// ---------------------------------------------------------------------------
// 3. The agent loop: ask Qwen what to do, do it, feed the result back.
//    Uses OpenAI-style chat.completions with function-calling.
// ---------------------------------------------------------------------------
async function runAgent(goal) {
  const headless = process.env.HEADLESS !== "false";
  const browser = await chromium.launch({ headless });
  const page = await browser.newPage();

  const messages = [
    {
      role: "system",
      content:
        "You control a web browser via tools. Start by navigating somewhere useful, then inspect the page with read_page before clicking or typing, since selectors on real pages vary. Call \"finish\" once you have the answer.",
    },
    { role: "user", content: `Goal: ${goal}` },
  ];

  for (let step = 0; step < MAX_STEPS; step++) {
    const response = await client.chat.completions.create({
      model: MODEL,
      messages,
      tools,
      tool_choice: "auto",
    });

    const msg = response.choices[0].message;
    messages.push(msg);

    const toolCall = msg.tool_calls?.[0];
    if (!toolCall) {
      console.log("\nQwen replied without calling a tool:\n", msg.content);
      break;
    }

    const name = toolCall.function.name;
    const input = JSON.parse(toolCall.function.arguments || "{}");

    if (name === "finish") {
      console.log("\n✅ Task complete:\n", input.result);
      try {
        mkdirSync("run-output", { recursive: true });
        const shot = await page.screenshot({ type: "png" });
        writeFileSync("run-output/final-screenshot.png", shot);
        writeFileSync("run-output/result.txt", input.result);
      } catch (err) {
        console.warn("Could not save run output:", err.message);
      }
      break;
    }

    console.log(`\n→ Step ${step + 1}: ${name}(${JSON.stringify(input)})`);

    let result;
    try {
      result = await executeTool(page, name, input);
    } catch (err) {
      result = { error: err.message };
    }

    // Feed the result back as a tool message so Qwen can decide the next step.
    messages.push({
      role: "tool",
      tool_call_id: toolCall.id,
      content: JSON.stringify(result),
    });
  }

  await browser.close();
}

const goal =
  process.argv.slice(2).join(" ") ||
  "Go to wikipedia.org and tell me today's featured article title.";
runAgent(goal).catch((err) => {
  console.error("Agent failed:", err);
  process.exit(1);
});
