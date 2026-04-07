import Link from "next/link";
import { createClient } from "@/lib/supabase/server";

export default async function Home() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <div className="max-w-md w-full space-y-8 text-center">
        <h1 className="text-4xl font-bold tracking-tight text-gray-900">
          {"{{PROJECT_NAME}}"}
        </h1>
        <p className="text-lg text-gray-600">
          Built with Next.js, Supabase, and Stripe
        </p>
        <div className="flex flex-col gap-4">
          {user ? (
            <>
              <p className="text-sm text-gray-500">
                Signed in as {user.email}
              </p>
              <Link
                href="/dashboard"
                className="rounded-lg bg-indigo-600 px-4 py-2 text-white hover:bg-indigo-700"
              >
                Go to Dashboard
              </Link>
            </>
          ) : (
            <>
              <Link
                href="/login"
                className="rounded-lg bg-indigo-600 px-4 py-2 text-white hover:bg-indigo-700"
              >
                Sign In
              </Link>
              <Link
                href="/signup"
                className="rounded-lg border border-gray-300 px-4 py-2 text-gray-700 hover:bg-gray-50"
              >
                Create Account
              </Link>
            </>
          )}
        </div>
      </div>
    </main>
  );
}
