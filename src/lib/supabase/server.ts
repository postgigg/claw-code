// src/lib/supabase/server.ts
import { createClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';

export async function createClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  return createClient(supabaseUrl, supabaseAnonKey);
}

export async function checkSession() {
  const { data, error } = await createClient().auth.getSession(cookies());
  if (!data.session) {
    return NextResponse.json({ error: 'Session expired' }, { status: 401 });
  }
  return null;
}
